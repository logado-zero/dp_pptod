import math, logging, copy, json
from collections import Counter, OrderedDict
from nltk.util import ngrams
import ontology


class MultiWozEvaluator(object):
    def __init__(self, reader, cfg):
        self.reader = reader
        self.domains = ontology.all_domains
        self.domain_files = self.reader.domain_files
        self.all_data = self.reader.data
        self.test_data = self.reader.test
        self.cfg = cfg
        self.set_attribute()


        self.all_info_slot = []
        for d, s_list in ontology.informable_slots.items():
            for s in s_list:
                self.all_info_slot.append(d+'-'+s)

        # only evaluate these slots for dialog success
        self.requestables = ['phone', 'address', 'postcode', 'reference', 'id']
        self.data_prefix = self.cfg.data_prefix
        self.mapping_pair_path = self.data_prefix + '/multi-woz/mapping.pair'


    def pack_dial(self, data):
        dials = {}
        for turn in data:
            dial_id = turn['dial_id']
            if dial_id not in dials:
                dials[dial_id] = []
            dials[dial_id].append(turn)
        return dials


    def run_metrics(self, data):
        if 'all' in self.cfg.exp_domains:
            metric_results = []
            metric_result = self._get_metric_results(data)
            metric_results.append(metric_result)

            if self.cfg.eval_per_domain:
                # all domain experiments, sub domain evaluation
                domains = [d+'_single' for d in ontology.all_domains]
                domains = domains + ['restaurant_train', 'restaurant_hotel','restaurant_attraction', 'hotel_train', 
                'hotel_attraction', 'attraction_train', 'restaurant_hotel_taxi', 'restaurant_attraction_taxi', 
                'hotel_attraction_taxi', ]
                for domain in domains:
                    file_list = self.domain_files.get(domain, [])
                    if not file_list:
                        print('No sub domain [%s]'%domain)
                    metric_result = self._get_metric_results(data, domain, file_list)
                    if metric_result:
                        metric_results.append(metric_result)
        else:
            # sub domain experiments
            metric_results = []
            for domain, file_list in self.domain_files.items():
                if domain not in self.cfg.exp_domains:
                    continue
                metric_result = self._get_metric_results(data, domain, file_list)
                if metric_result:
                    metric_results.append(metric_result)

        return metric_results


    def _get_metric_results(self, data, domain='all', file_list=None):
        metric_result = {'domain': domain}
        

        act_precision, act_recall, act_f1, dial_num = self.aspn_eval(data, file_list)
        accu_single_dom, accu_multi_dom, multi_dom_num = self.domain_eval(data, file_list)

        

        
        metric_result.update({'act_f1':act_f1,'act_precision':act_precision, 'act_recall':act_recall,
                                    'dial_num': dial_num, 'accu_single_dom': accu_single_dom, 'accu_multi_dom': accu_multi_dom})
        if domain == 'all':
            logging.info('-------------------------- All DOMAINS --------------------------')
        else:
            logging.info('-------------------------- %s (# %d) -------------------------- '%(domain.upper(), dial_num))
        
        logging.info('[DP] act precision:%2.1f  act recall: %2.1f   act f1: %2.1f'%(act_precision, act_recall, act_f1))
            
        
        
        logging.info('[DOM] accuracy: single %2.1f / multi: %2.1f (%d)'%(accu_single_dom, accu_multi_dom, multi_dom_num))
        
        return metric_result
        

    def set_attribute(self):
        self.cfg.use_true_prev_bspn = True
        self.cfg.use_true_prev_aspn = True
        self.cfg.use_true_db_pointer = True
        self.cfg.use_true_prev_resp = True
        self.cfg.use_true_pv_resp = True
        self.cfg.same_eval_as_cambridge = True
        self.cfg.use_true_domain_for_ctr_eval = True
        self.cfg.use_true_bspn_for_ctr_eval = False
        self.cfg.use_true_curr_bspn = False
        self.cfg.use_true_curr_aspn = False
        self.cfg.same_eval_act_f1_as_hdsa = False
        self.cfg.eval_per_domain = False

    def value_similar(self, a,b):
        return True if a==b else False
        # the value equal condition used in "Sequicity" is too loose
        if a in b or b in a or a.split()[0] == b.split()[0] or a.split()[-1] == b.split()[-1]:
            return True
        return False

    def _bspn_to_dict(self, bspn, no_name=False, no_book=False, bspn_mode = 'bspn'):
        constraint_dict = self.reader.bspan_to_constraint_dict(bspn, bspn_mode = bspn_mode)
        constraint_dict_flat = {}
        for domain, cons in constraint_dict.items():
            for s,v in cons.items():
                key = domain+'-'+s
                if no_name and s == 'name':
                    continue
                if no_book:
                    if s in ['people', 'stay'] or key in ['hotel-day', 'restaurant-day','restaurant-time'] :
                        continue
                constraint_dict_flat[key] = v
        return constraint_dict_flat

    def _constraint_compare(self, truth_cons, gen_cons, slot_appear_num=None, slot_correct_num=None):
        tp,fp,fn = 0,0,0
        false_slot = []
        for slot in gen_cons:
            v_gen = gen_cons[slot]
            if slot in truth_cons and self.value_similar(v_gen, truth_cons[slot]):  #v_truth = truth_cons[slot]
                tp += 1
                if slot_correct_num is not None:
                    slot_correct_num[slot] = 1 if not slot_correct_num.get(slot) else slot_correct_num.get(slot)+1
            else:
                fp += 1
                false_slot.append(slot)
        for slot in truth_cons:
            v_truth = truth_cons[slot]
            if slot_appear_num is not None:
                slot_appear_num[slot] = 1 if not slot_appear_num.get(slot) else slot_appear_num.get(slot)+1
            if slot not in gen_cons or not self.value_similar(v_truth, gen_cons[slot]):
                fn += 1
                false_slot.append(slot)
        acc = len(self.all_info_slot) - fp - fn
        return tp,fp,fn, acc, list(set(false_slot))

    def domain_eval(self, data, eval_dial_list = None):
        dials = self.pack_dial(data)
        corr_single, total_single, corr_multi, total_multi = 0, 0, 0, 0

        dial_num = 0
        for dial_id in dials:
            if eval_dial_list and dial_id+'.json' not in eval_dial_list:
                continue
            dial_num += 1
            dial = dials[dial_id]
            wrong_pred = []

            prev_turn_domain = ['general']

            for turn_num, turn in enumerate(dial):
                if turn_num == 0:
                    continue
                true_domains = self.reader.dspan_to_domain(turn['dspn'])
                if self.cfg.enable_dspn:
                    pred_domains = self.reader.dspan_to_domain(turn['dspn_gen'])
                else:
                    aspn = 'aspn' if not self.cfg.enable_aspn else 'aspn_gen'
                    turn_dom_da = []
                    for a in turn[aspn].split():
                        if a[1:-1] in ontology.all_domains + ['general']:
                            turn_dom_da.append(a[1:-1])

                    # get turn domain
                    turn_domain = []
                    for dom in turn_dom_da:
                        if dom != 'booking' and dom not in turn_domain:
                            turn_domain.append(dom)
                    if not turn_domain:
                        turn_domain = prev_turn_domain
                    if len(turn_domain) == 2 and 'general' in turn_domain:
                        turn_domain.remove('general')
                    if len(turn_domain) == 2:
                        if len(prev_turn_domain) == 1 and prev_turn_domain[0] == turn_domain[1]:
                            turn_domain = turn_domain[::-1]
                    prev_turn_domain = copy.deepcopy(turn_domain)


                    turn['dspn_gen'] = ' '.join(['['+d+']' for d in turn_domain])
                    pred_domains = {}
                    for d in turn_domain:
                        pred_domains['['+d+']'] = 1

                if len(true_domains) == 1:
                    total_single += 1
                    if pred_domains == true_domains:
                        corr_single += 1
                    else:
                        wrong_pred.append(str(turn['turn_num']))
                        turn['wrong_domain'] = 'x'
                else:
                    total_multi += 1
                    if pred_domains == true_domains:
                        corr_multi += 1
                    else:
                        wrong_pred.append(str(turn['turn_num']))
                        turn['wrong_domain'] = 'x'

            # dialog inform metric record
            dial[0]['wrong_domain'] = ' '.join(wrong_pred)
        accu_single = corr_single / (total_single + 1e-10)
        accu_multi = corr_multi / (total_multi + 1e-10)
        return accu_single * 100, accu_multi * 100, total_multi



    def aspn_eval(self, data, eval_dial_list = None):

        def _get_tp_fp_fn(label_list, pred_list):
            tp = len([t for t in pred_list if t in label_list])
            fp = max(0, len(pred_list) - tp)
            fn = max(0, len(label_list) - tp)
            return tp, fp, fn

        dials = self.pack_dial(data)
        total_tp, total_fp, total_fn = 0, 0, 0

        dial_num = 0
        for dial_id in dials:
            if eval_dial_list and dial_id+'.json' not in eval_dial_list:
                continue
            dial_num += 1
            dial = dials[dial_id]
            wrong_act = []
            for turn_num, turn in enumerate(dial):
                if turn_num == 0:
                    continue
                if self.cfg.same_eval_act_f1_as_hdsa:
                    pred_acts, true_acts = {}, {}
                    for t in turn['aspn_gen']:
                        pred_acts[t] = 1
                    for t in  turn['aspn']:
                        true_acts[t] = 1
                    tp, fp, fn = _get_tp_fp_fn(true_acts, pred_acts)
                else:
                    pred_acts = self.reader.aspan_to_act_list(turn['aspn_gen'])
                    true_acts = self.reader.aspan_to_act_list(turn['aspn'])
                    tp, fp, fn = _get_tp_fp_fn(true_acts, pred_acts)
                if fp + fn !=0:
                    wrong_act.append(str(turn['turn_num']))
                    turn['wrong_act'] = 'x'

                total_tp += tp
                total_fp += fp
                total_fn += fn

            dial[0]['wrong_act'] = ' '.join(wrong_act)
        precision = total_tp / (total_tp + total_fp + 1e-10)
        recall = total_tp / (total_tp + total_fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        return precision*100, recall*100, f1 * 100, dial_num






if __name__ == '__main__':
    pass
