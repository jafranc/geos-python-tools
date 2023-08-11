import numpy as np
import matplotlib.pyplot as plt
import re
import pandas as pd
import argparse


class LogParser:
    def process(self, fname):
        buffer, time_tag, _ = self._split_log_(fname)
        self.MB = Mass_balances()
        values = pd.DataFrame(columns=self.MB.collabels)

        for i, bits in enumerate(buffer):
            new_row = {}
            new_row['time'] = time_tag[i + 1]
            for key, exp in self.MB.expression.items():
                value = self.MB.process(bits, exp)
                if value is not None:
                    if isinstance(value, float):
                        new_row[key] = value
                    else:
                        for icol in range(value.shape[0]):
                            new_row[key + '_{}'.format(icol)] = value[icol]
            if len(new_row) == len(values.columns):
                values = pd.concat(
                    [pd.DataFrame(new_row, columns=values.columns, index=[i]), values.loc[:]]).reset_index(
                    drop=True)

        self._plot_dataframe_(values, 'time',
                              ['mobile_0', 'immobile_0', 'dissolved_2', 'mobile_0+immobile_0+dissolved_2'])

    def _plot_dataframe_(self, values, xkey, ykeys):
        sec2day = 60 * 60 * 24
        kg2Mt = 1e3 * 1e6

        for k in ykeys:
            yfield = 0 * values['time']
            if re.findall(r'\+', k):
                subkeys = k.split('+')
                for sk in subkeys:
                    yfield += values[sk]
            else:
                yfield = values[k]

            plt.plot(values[xkey].to_numpy() / sec2day, yfield.to_numpy() / kg2Mt, '-+', label=k)
        plt.legend()
        plt.show()

    # todo  suppport complex operations
    # def __operation_on_key__(values, keys):
    #
    #     for k in keys:
    #         if re.match(r'\*',k):
    #             __operation_on_key__(values, k.split('*'))
    #         elif re.match(r'\/',k):
    #             __operation_on_key__(values, k.split('/'))
    #         elif re.match(r'\+',k):
    #             __operation_on_key__(values,k.split('+'))
    #         elif re.match(r'-',k):
    #             __operation_on_key__(values,k.split('-'))

    def _split_log_(self, fname):
        buffer = []
        time_tag = {}
        buffer_ = ''
        with open(fname) as f:
            for line in f:
                if re.match(r'^Time', line):
                    buffer.append(buffer_)
                    time_tag[len(buffer)] = float(re.match(r'^Time: (\d+\.\d+e[+-]\d+)', line).group(1))
                    buffer_ = ''
                elif re.match(r'Cleaning up events', line):
                    buffer.append(buffer_)
                else:
                    buffer_ += line
        header = buffer.pop(0)  # discard pre-simulation values
        return buffer, time_tag, header


class Mass_balances:
    """
        Class design to handle mass info grep-ing and produce a DataFrame from it.
    """

    def __init__(self):
        self.expression = {}
        universal_float = r'\d+(\.)?(\d+)?(e[+-]\d+)?'
        self.expression[
            'total'] = r'reservoir[1-9]: Phase mass: \{ (' + universal_float + '), (' + universal_float + ') \} kg'
        self.expression[
            'trapped'] = r'reservoir[1-9]: Trapped phase mass \(metric 1\): \{ (' + universal_float + '), (' + universal_float + ') \} kg'
        self.expression[
            'immobile'] = r'reservoir[1-9]: Immobile phase mass \(metric 2\): \{ (' + universal_float + '), (' + universal_float + ') \} kg'
        self.expression[
            'mobile'] = r'reservoir[1-9]: Mobile phase mass \(metric 2\): \{ (' + universal_float + '), (' + universal_float + ') \} kg'
        self.expression[
            'dissolved'] = r'reservoir[1-9]: Dissolved component mass: \{ \{ (' + universal_float + '), (' + universal_float + ') \}, \{ (' + universal_float + '), (' + universal_float + ') \} \} kg'

        self.collabels = list(self.expression.keys())
        self.collabels = self._expand_labels_(self.collabels)
        self.collabels.insert(0, 'time')

    def process(self, buffer, exp):
        trapped = []
        t = re.findall(exp, buffer)
        if len(t) > 0:
            for matched in t:
                trapped.append([float(matched[i]) for i in range(0, len(matched), 4)])
        else:
            return None
        return np.sum(np.asarray(trapped), axis=0)

    def _expand_labels_(self, labels):
        new_labels = []
        for i, label in enumerate(labels):
            if re.match(r'total|immobile|mobile|trapped', label):
                new_labels.extend([label + '_0', label + '_1'])
            elif re.match(r'dissolved', label):
                # todo read nphase and ncp in header bit
                new_labels.extend([label + '_{}'.format(xc) for xc in range(4)])
            else:
                new_labels.append(label)

        return new_labels


if __name__ == "__main__":
    descr = 'Set of python script for post-processing GEOS\n'
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument('--log', metavar='logfile', nargs=1,
                        help='path to the logfile')


    args = parser.parse_args()

    if args.log:
        parser = LogParser()
        parser.process(fname=args.log[0])
        # parser.process(fname='/mnt/droplet/slurm-27086382.out')
    else:
        parser.print_help()
