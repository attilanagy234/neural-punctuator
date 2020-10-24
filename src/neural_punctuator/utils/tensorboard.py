

def print_metrics(counter,
                  loss,
                  summary_writer=None,
                  prefix=None,
                  model_name=""):

    print(
        prefix+"\tloss: {0:.5f}".format(
            loss
        ))

    if summary_writer is not None:
        assert prefix is not None
        summary_writer.add_scalar(f'{model_name}_{prefix}_loss', loss, counter)
