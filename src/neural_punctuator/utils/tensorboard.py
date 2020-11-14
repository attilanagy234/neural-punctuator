

def print_metrics(counter,
                  metrics,
                  summary_writer=None,
                  prefix=None,
                  model_name=""):

    if 'cls_report' in metrics.keys():
        metrics = metrics.copy()
        del metrics['cls_report']

    # print(prefix + "\t" + "\t".join([f"{key}:{value:.5f}" for key, value in metrics.items()]))

    if summary_writer is not None:
        assert prefix is not None
        for key, value in metrics.items():
            summary_writer.add_scalar(f'{model_name}_{prefix}_{key}', value, counter)

