import numpy as np
import itertools as itt


class StateProposals(object):
    def __init__(self, cvs, positive_trajs, negative_trajs,
                 max_combinations=None, cutoff=0.35, cutoff_step=0.05,
                 cutoff_max=0.45):
        self.cvs = cvs
        self.name_to_cv = {}
        for cv in cvs:
            self.name_to_cv[cv.name] = cv
        self.positive_trajs = positive_trajs
        self.negative_trajs = negative_trajs
        self.cutoff = cutoff
        self.cutoff_step = cutoff_step
        self.cutoff_max = cutoff_max
        if max_combinations is None:
            self.max_combinations = len(cvs)
        else:
            self.max_combinations = max_combinations
        self.pos_cv_results = self.make_cv_result_dict(cvs, positive_trajs)
        self.neg_cv_results = self.make_cv_result_dict(cvs, negative_trajs)
        self.cutoff_list = self.make_cutoff_list(cvs, cutoff, cutoff_step,
                                                 cutoff_max)
        self.cv_combinations_dict = self.make_cv_combinations(cvs,
                                                              max_combinations)
        self.pos_result_dict = self.make_result_dict(self.cv_combinations_dict,
                                                     self.pos_cv_results,
                                                     self.cutoff_list,
                                                     cvs)

        self.neg_result_dict = self.make_result_dict(self.cv_combinations_dict,
                                                     self.neg_cv_results,
                                                     self.cutoff_list,
                                                     cvs)
        self.total_result_dict = self.make_total_result_dict(
                                                self.pos_result_dict,
                                                self.neg_result_dict,
                                                len(positive_trajs),
                                                len(negative_trajs))
        self.output_dict = self.make_output_dict(self.total_result_dict)

    def make_cv_combination_key(self, cv_combination):
        key = ""
        if type(cv_combination) != list:
            cv_combination = [cv_combination]
        for comb in cv_combination:
            if type(comb) == list:
                key += self.make_cv_combination_key(comb)
            elif type(comb) == str:
                key += " " + comb + " "
            else:
                key += comb.name
        return key

    def make_cutoff_list(self, cvs, cutoff, cutoff_step, cutoff_max):
        max_decimal = max([len(str(i).split('.', 1)[1])
                           for i in [cutoff, cutoff_step, cutoff_max]])
        cutoff_options = [round(cutoff+i*cutoff_step, max_decimal) for i in
                          range(int((cutoff_max - cutoff) / cutoff_step) + 1)]
        combined_cutoff_options = [i for i in itt.product(cutoff_options,
                                                          repeat=len(cvs))]
        cutoff_list = [self.make_cutoff_dict(cvs, i)
                       for i in combined_cutoff_options]
        return cutoff_list

    def total_length(self, n_list):
        try:
            n_list = n_list[0]
        except (IndexError, TypeError):
            return 0
        try:
            return sum([self.total_length([i]) for i in n_list])
        except TypeError:
            return 1
    def make_cutoff_dict(self, cv_list, cutoff_list):
        return_dict = {}
        for i, cv in enumerate(cv_list):
            return_dict[cv] = cutoff_list[i]
        return return_dict

    def make_cv_result_dict(self, cv_list, trajs):
        result_dict = {}
        max_traj = max([len(traj) for traj in trajs])
        for cv in cv_list:
            result_array = np.zeros(shape=(len(trajs), max_traj))
            for i_traj, traj in enumerate(trajs):
                try:
                    result_cv = np.array([i[0] for i in cv(traj)])
                except (TypeError, IndexError):  # either scalar, float or int
                    result_cv = np.array([i for i in cv(traj)])
                for i_array in range(max_traj):
                    try:
                        result_array[i_traj][i_array] = result_cv[i_array]
                    except IndexError:
                        result_array[i_traj][i_array] = np.inf
            result_dict[cv] = result_array
        return result_dict

    def combine_cv_and_or(self, cv_list, and_or_list, and_or_index=0):
        result = []
        if len(cv_list) == 1:
            result.append(cv_list)
            result.append(0)
            return result

        for cv_index, cv in enumerate(cv_list):
            try:
                cv[0]
            except TypeError:
                result.append(cv)
                if cv_index != len(cv_list)-1:
                    result.append(and_or_list[and_or_index])
                    and_or_index += 1
            else:
                temp = []
                for i in self.combine_cv_and_or(cv, and_or_list, and_or_index):
                    temp.append(i)
                    and_or_index = temp[-1]

                result.append(temp[:-1])

            if and_or_index != len(and_or_list):
                result.append(and_or_index)

        return result

    def add_combination(self, result_dict,
                        current_combination,
                        possible_length):
        combination_list = []
        for j in range(1, possible_length + 1):
            print possible_length

            added_combination = result_dict[j]['cv_combinations']
            print added_combination
            itt_product = itt.product(current_combination,
                                      added_combination)
            combination_list.append([i for i in itt_product
                                     if i[0] != i[1]])
            unique_list = []
            for combination in combination_list:
                current_combination = []
                for cv in combination:
                    appendable = True
                    try:
                        if cv[0] in cv[1]:
                            appendable = False
                    except TypeError:
                        pass
                    try:
                        if cv[1] in cv[0]:
                            appendable = False
                    except TypeError:
                        pass
                    if appendable:
                        current_combination.append(cv)
                unique_list.append(current_combination)
        return unique_list

    def next_combination(self, result_dict, current_combination, max_length):
        current_length = self.total_length(current_combination)
        if current_length == 0:
            current_combination =[i for i in result_dict[1]['cv_combinations']]
            current_length = self.total_length(current_combination)
        possible_length = max_length - current_length
        if possible_length == 0:
            return current_combination
        else:
            unique_list = self.add_combination(result_dict, current_combination,
                                               possible_length)
            results = [self.next_combination(result_dict, i, max_length)
                       for i in unique_list]
            flattend = [val for sublist in results for val in sublist]
            unique_results = []
            reversed_results = []
            for result in flattend:
                if result[0] != result[1]:
                    if (result not in unique_results and
                       result not in reversed_results):
                        unique_results.append(result)
                        reversed_results.append(result[::-1])
            return unique_results

    def make_cv_combinations(self, cv_list, max_combinations=None):
        if max_combinations is None:
            max_combinations = len(cv_list)
        and_or_list = ['and', 'or']
        result_dict = {1: {'cv_combinations': cv_list,
                           'and_or_combinations': [],
                           'cv_and_or_combinations': cv_list,
                           'total_list': cv_list}}
        for j in range(2, max_combinations + 1):
            cv_combinations = self.next_combination(result_dict, [], j)
            try:
                cv_combinations[0]
            except TypeError:
                cv_combinations = [cv_combinations]
                cv_combinations = [cv_combinations]
            and_or_combinations = [i for i in itt.product(and_or_list,
                                                          repeat=j-1)]
            cv_and_or_combinations = [i for i in itt.product(cv_combinations,
                                                             and_or_combinations
                                                             )]
            total_list = [self.combine_cv_and_or(i[0], i[1])
                          for i in cv_and_or_combinations]
            result_dict[j] = {'cv_combinations': cv_combinations,
                              'and_or_combinations': and_or_combinations,
                              'cv_and_or_combinations': cv_and_or_combinations,
                              'total_list': total_list}
        return result_dict

    def single_mask(self, cv_array, cutoff):
        return np.where(cv_array < cutoff, 1, 0)

    def or_mask(self, mask1, mask2):
        return mask1+mask2

    def and_mask(self, mask1, mask2):
        return np.multiply(mask1, mask2)

    def combination_to_mask(self, cv_combination, cv_result_dict, cutoff_dict):
        total_mask = None
        if type(cv_combination) != list:
            cv = cv_combination
            return self.single_mask(cv_result_dict[cv], cutoff_dict[cv])
        set_or = False
        set_and = False
        for cv in cv_combination:
            mask = None
            if type(cv) == list:
                mask = self.combination_to_mask(cv, cv_result_dict, cutoff_dict)
            elif type(cv) != str:
                mask = self.single_mask(cv_result_dict[cv], cutoff_dict[cv])
            elif cv == 'or':
                set_or = True
            elif cv == 'and':
                set_and = True
            if set_or and mask is not None:
                total_mask = self.or_mask(total_mask, mask)
                set_or = False
            elif set_and and mask is not None:
                total_mask = self.and_mask(total_mask, mask)
            if total_mask is None:
                total_mask = mask
        return total_mask

    def make_result_dict(self, cv_combination_dict, cv_result_dict,
                         cutoff_list, cv_list):
        result_dict = {}
        for combination_type in cv_combination_dict.keys():
            c_type_dict = {}
            total_list = cv_combination_dict[combination_type]['total_list']
            for cv_combination in total_list:
                internal_cv_combination_dict = {}
                cv_key = self.make_cv_combination_key(cv_combination)
                for cutoff_dict in cutoff_list:
                    try:
                        cutoff_key = str([cutoff_dict[cv] for cv in cv_list
                                          if cv in cv_combination])
                    except TypeError:
                        cutoff_key = str([cutoff_dict[cv_combination]])
                    result_mask = self.combination_to_mask(cv_combination,
                                                           cv_result_dict,
                                                           cutoff_dict)
                    result = []
                    for i in range(result_mask.shape[0]):
                        if sum(result_mask[i]) > 0:
                            result.append(i)
                    internal_cv_combination_dict[cutoff_key] = result
                c_type_dict[cv_key] = internal_cv_combination_dict
            result_dict[combination_type] = c_type_dict
        return result_dict

    def make_total_result_dict(self, pos_dict, neg_dict, pos_max, neg_max):
        result_dict = {}
        for c_type in pos_dict.keys():
            c_type_dict = {}
            for cv_comb in pos_dict[c_type].keys():
                cv_comb_dict = {}
                for cutoff_comb in pos_dict[c_type][cv_comb].keys():
                    n_pos = len(pos_dict[c_type][cv_comb][cutoff_comb])
                    n_neg = len(neg_dict[c_type][cv_comb][cutoff_comb])
                    procent_pos = float(n_pos)/pos_max
                    procent_neg = float(n_neg)/neg_max
                    try:
                        absolute_fraction = float(n_pos)/n_neg
                    except ZeroDivisionError:
                        if n_pos > 0:
                            absolute_fraction = np.inf
                        else:
                            absolute_fraction = 0.0
                    try:
                        relative_fraction = procent_pos/procent_neg
                    except ZeroDivisionError:
                        if procent_pos > 0:
                            relative_fraction = np.inf
                        else:
                            relative_fraction = 0.0
                    cutoff_comb_dict = {'n_pos': n_pos,
                                        'n_neg': n_neg,
                                        'absolute_fraction': absolute_fraction,
                                        'relative_fraction': relative_fraction}
                    cv_comb_dict[cutoff_comb] = cutoff_comb_dict
                c_type_dict[cv_comb] = cv_comb_dict
            result_dict[c_type] = c_type_dict
        return result_dict

    def ctype_output(self, c_type, total_result_dict):
        ctype_max_n_pos = 0
        ctype_max_n_pos_list = []
        ctype_min_n_neg = np.inf
        ctype_min_n_neg_list = []
        ctype_abs_fraction = 0
        ctype_abs_fraction_list = []
        ctype_rel_fraction = 0
        ctype_rel_fraction_list = []
        for cv_comb in total_result_dict[c_type].keys():
            for cutoff_comb in total_result_dict[c_type][cv_comb].keys():
                result = total_result_dict[c_type][cv_comb][cutoff_comb]
                if result['n_pos'] > ctype_max_n_pos:
                    ctype_max_n_pos_list = [cv_comb+cutoff_comb]
                    ctype_max_n_pos = result['n_pos']
                elif result['n_pos'] == ctype_max_n_pos:
                    ctype_max_n_pos_list.append(cv_comb+cutoff_comb)

                if result['n_neg'] < ctype_min_n_neg:
                    ctype_min_n_neg_list = [cv_comb+cutoff_comb]
                    ctype_min_n_neg = result['n_neg']
                elif result['n_neg'] == ctype_min_n_neg:
                    ctype_min_n_neg_list.append(cv_comb+cutoff_comb)

                if result['absolute_fraction'] > ctype_abs_fraction:
                    ctype_abs_fraction_list = [cv_comb+cutoff_comb]
                    ctype_abs_fraction = result['absolute_fraction']
                elif result['absolute_fraction'] == ctype_abs_fraction:
                    ctype_abs_fraction_list.append(cv_comb+cutoff_comb)

                if result['relative_fraction'] > ctype_rel_fraction:
                    ctype_rel_fraction_list = [cv_comb+cutoff_comb]
                    ctype_rel_fraction = result['relative_fraction']
                elif result['relative_fraction'] == ctype_rel_fraction:
                    ctype_rel_fraction_list.append(cv_comb+cutoff_comb)

        ctype_dict = {'max_n_pos': ctype_max_n_pos,
                      'max_n_pos_list': ctype_max_n_pos_list,
                      'min_n_neg': ctype_min_n_neg,
                      'min_n_neg_list': ctype_min_n_neg_list,
                      'abs_fraction': ctype_abs_fraction,
                      'abs_fraction_list': ctype_abs_fraction_list,
                      'rel_fraction': ctype_rel_fraction,
                      'rel_fraction_list': ctype_rel_fraction_list}
        return ctype_dict

    def make_output_dict(self, total_result_dict):
        output_dict = {}
        for c_type in total_result_dict.keys():
            output_dict[c_type] = self.ctype_output(c_type, total_result_dict)
        total_max_n_pos = 0
        total_max_n_pos_list = []
        total_min_n_neg = np.inf
        total_min_n_neg_list = []
        total_abs_fraction = 0
        total_abs_fraction_list = []
        total_rel_fraction = 0
        total_rel_fraction_list = []
        for ctype in output_dict.keys():
            ctype_dict = output_dict[ctype]
            if ctype_dict['max_n_pos'] > total_max_n_pos:
                total_max_n_pos = ctype_dict['max_n_pos']
                total_max_n_pos_list = [i for i in ctype_dict['max_n_pos_list']]
            elif ctype_dict['max_n_pos'] == total_max_n_pos:
                total_max_n_pos_list.extend(ctype_dict['max_n_pos_list'])

            if ctype_dict['min_n_neg'] < total_min_n_neg:
                total_min_n_neg = ctype_dict['min_n_neg']
                total_min_n_neg_list = [i for i in ctype_dict['min_n_neg_list']]
            elif ctype_dict['min_n_neg'] == total_min_n_neg:
                total_min_n_neg_list.extend(ctype_dict['min_n_neg_list'])

            if ctype_dict['abs_fraction'] > total_abs_fraction:
                total_abs_fraction = ctype_dict['abs_fraction']
                total_abs_fraction_list = [i for i in
                                           ctype_dict['abs_fraction_list']]
            elif ctype_dict['abs_fraction'] == total_abs_fraction:
                total_abs_fraction_list.extend(ctype_dict['abs_fraction_list'])

            if ctype_dict['rel_fraction'] > total_rel_fraction:
                total_rel_fraction = ctype_dict['rel_fraction']
                total_rel_fraction_list = [i for i in
                                           ctype_dict['rel_fraction_list']]
            elif ctype_dict['rel_fraction'] == total_rel_fraction:
                total_rel_fraction_list.extend(ctype_dict['rel_fraction_list'])

        output_dict['total'] = {'max_n_pos': total_max_n_pos,
                                'max_n_pos_list': total_max_n_pos_list,
                                'min_n_neg': total_min_n_neg,
                                'min_n_neg_list': total_min_n_neg_list,
                                'abs_fraction': total_abs_fraction,
                                'abs_fraction_list': total_abs_fraction_list,
                                'rel_fraction': total_rel_fraction,
                                'rel_fraction_list': total_rel_fraction_list}
        print output_dict[1]
        return output_dict
