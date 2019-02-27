import os
import itertools
import threading
import numpy as np


optim_dir = 'optim/'

def get_optim_path(model):
    return os.path.join(optim_dir, model)

def is_calculated(model):
    return os.path.isfile('pickles/' + get_optim_path(model))

def get_scores(model):
    assert is_calculated(model)
    return pload(get_optim_path(model))

def save_scores(scores, model):
    psave(scores, get_optim_path(model))

def remove_model(model):
    if is_calculated(model):
        os.remove('pickles/' + optim_dir + model)

def cleanup():
    global my_models
    for item in os.listdir('pickles/optim/'):
        if item not in my_models:
            os.remove('pickles/optim/' + item)

def log(message, logfile, display=True):
    if display:
        print(message)
    if logfile is not None:
        f = open(logfile, 'a')
        f.write(message)
        if len(message) < 1 or message[-1] != '\n':
            f.write('\n')
        f.close()

def auto_ensemble(pool, metric, logfile=None, **args):
    # If pool is a set, then choose the best option
    if type(pool) == set:
        items = []
        for pool_item in pool:
            opt = optimize(pool_item, metric, logfile, **args)
            items.append((metric(get_scores(opt)), opt))
        items = sorted(items)
        return items[-1][1]
    
    # If pool is a list, calculate each pool item
    for i in range(len(pool)):
        if type(pool[i]) != str:
            pool = pool[:i] + (optimize(pool[i], metric, logfile, **args),) + pool[i+1:]
    
    log("Optimizing pool: {}.".format(', '.join(pool)), logfile)
    for item in pool:
        assert is_calculated(item)
    
    # Shuffling or sorting pool by metric if needed
    order = args.get('order', 'fixed')
    if order == 'sorted':
        log("Sorting by metric...", logfile)
        new_pool = []
        for name in pool:
            scores = get_scores(name)
            metric_val = metric(scores)
            log("{}:\t{}".format(name, metric_val), logfile)
            new_pool.append((metric_val, name))
        new_pool = sorted(new_pool)
        new_pool = [elem[1] for elem in new_pool]
        log("Sorted order: {}.".format(', '.join(new_pool)), logfile)
        pool = new_pool
    elif order == 'random':
        log("Shuffling pool...", logfile)
        seed = args.get('random_state', None)
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(pool)
    
    lin_coeffs = args['lin_coeffs']
    enable_power = args['enable_power']
    lin_delta = args['lin_delta']
    pow_coeffs_gamma = args['pow_coeffs_gamma']
    pow_coeffs_delta = args['pow_coeffs_delta']
    pow_delta_gamma = args['pow_delta_gamma']
    pow_delta_delta = args['pow_delta_delta']
    prepr_funcs = args.get('prepr_funcs', [lambda x: x])
    num_iters = args['num_iters']
    n_jobs = args.get('n_jobs', 1)
    
    name1, name2 = pool[0], pool[1]
    merged_names = sort_seq("({}+{})".format(name1, name2))

    if is_calculated(merged_names):
        log('Blend {} already exists. Optimal parameters are {}.'.format(merged_names, best_params[merged_names]), logfile)
        
        if args.get('cleanup', True):
            remove_model(name1)
            remove_model(name2)

        pool = (merged_names,) + pool[2:]
        if len(pool) == 1:
            return pool[0]
        return optimize(pool, metric, logfile, **args)
    
    sc1, sc2 = get_scores(name1), get_scores(name2)
    log('Blending {} and {}...'.format(name1, name2), logfile)
    log('Calculating models separately...', logfile)
    metric1 = metric(sc1)
    metric2 = metric(sc2)
    log('{}:\t{}\n{}:\t{}'.format(name1, metric1, name2, metric2), logfile)
    log('Getting the 1st level optimum...', logfile)
    best_metric, best_alpha, best_pows, best_name = None, None, None, None
    if metric1 > metric2:
        best_metric = metric1
        best_alpha = 1.
        best_pows = [1., 1.]
        best_name = name1
    else:
        best_metric = metric2
        best_alpha = 0.
        best_pows = [1., 1.]
        best_name = name2

    _lin_delta = lin_delta
    _pow_delta_g = pow_delta_gamma
    _pow_delta_d = pow_delta_delta
    _lin_coeffs = lin_coeffs.copy()
    _pow_coeffs_gamma = pow_coeffs_gamma.copy()
    _pow_coeffs_delta = pow_coeffs_delta.copy()

    def _iteration(_lin_coeffs, _pow_coeffs_gamma, _pow_coeffs_delta):
        nonlocal best_metric, best_alpha, best_pows, best_name, prepr_funcs
        
        params = list(itertools.product(_lin_coeffs, _pow_coeffs_gamma, _pow_coeffs_delta, list(np.arange(len(prepr_funcs)) + 1)))
        
        tasks = [[] for _ in range(n_jobs)]
        pointer = 0
        idx = 0
        while idx < len(params):
            tasks[pointer].append(params[idx])
            idx += 1
            pointer += 1
            if pointer >= n_jobs:
                pointer = 0
        
        output = [None] * len(tasks)
        threads = []
        thread_input = tasks

        def worker(data_idx):
            task = thread_input[data_idx]
            _best_metric = -1
            _best_alpha, _best_pows = None, None
            _best_name = None
            
            for item in task:
                alpha, gamma, delta, prepr_index = item
                prepr_func = prepr_funcs[prepr_index - 1]
                beta = 1 - alpha
                blend_result = blender.blend([sc1, prepr_func(sc2)], [alpha, beta], [gamma, delta])
                metric_val = metric(blend_result)
                merged_name = "{:.3f}*{}^{:.3f}+[{}]{:.3f}*{}^{:.3f}".format(alpha, name1, gamma, prepr_index, beta, name2, delta)
                log("{} metric: {:.5f}".format(merged_name, metric_val), logfile)
                if metric_val > _best_metric:
                    _best_alpha, _best_pows = alpha, (gamma, delta)
                    _best_name = merged_name
                    _best_metric = metric_val
            
            result = (_best_metric, _best_alpha, _best_pows, _best_name)
            output[data_idx] = result

        for i in range(len(tasks)):
            t = threading.Thread(target=worker, args=(i,), daemon=True)
            t.start()
            threads.append(t)

        for t in threads:
            if t is not None:
                t.join()
        
        # Gathering threads output
        for output_item in output:
            _best_metric, _best_alpha, _best_pows, _best_name = output_item
            if _best_metric > best_metric:
                best_metric, best_alpha, best_pows, best_name = _best_metric, _best_alpha, _best_pows, _best_name

    for iteration in range(1, num_iters + 1):
        log("\nStarting {}-level blend optimization...".format(iteration), logfile)
        log("Coeffs: {}".format((_lin_coeffs, _pow_coeffs_gamma, _pow_coeffs_delta)), logfile)
        _iteration(_lin_coeffs, _pow_coeffs_gamma, _pow_coeffs_delta)

        log("({}) Best is: {} (metric: {:.5f})".format(iteration, best_name, best_metric), logfile)
        _lin_delta /= 2.
        _pow_delta_g /= 2.
        _pow_delta_d /= 2.

        def segment(x, delta):
            return [x - delta, x, x + delta]

        _lin_coeffs = segment(best_alpha, _lin_delta)
        if enable_power:
            _pow_coeffs_gamma = segment(best_pows[0], _pow_delta_g)
            _pow_coeffs_delta = segment(best_pows[1], _pow_delta_d)

    log("\nOverall best: {}, metric: {}, params: ({}, {}, {}, {})".format(best_name, best_metric, best_alpha, 1 - best_alpha, best_pows[0], best_pows[1]), logfile)
    best_model = blender.blend([sc1, sc2], [best_alpha, 1 - best_alpha], [best_pows[0], best_pows[1]])
    best_params[merged_names] = (best_alpha, best_pows)
    save_scores(best_model, merged_names)

    if args.get('cleanup', True):
        remove_model(name1)
        remove_model(name2)

    pool = (merged_names,) + pool[2:]
    if len(pool) == 1:
        return pool[0]
    return optimize(pool, metric, logfile, **args)