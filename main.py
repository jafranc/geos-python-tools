import numpy as np
import matplotlib.pyplot as plt
import re
import pandas as pd
import argparse
import utils.expressions


class LogParser:

    def process(self, fnames):

        for name in fnames:
            self._process(name)

        plt.legend()
        plt.show()

    def _process(self, fname):
        buffer, time_tag, dt, _ = self._split_log_(fname)
        self.MB = Mass_balances()
        self.NI = Newton_iterations()
        # values = pd.DataFrame(columns=self.MB.collabels)
        values = pd.DataFrame(columns=self.NI.collabels)

        for i, bits in enumerate(buffer):
            new_row = {}
            new_row['time'] = time_tag[i + 1]
            new_row['dt'] = dt[i + 1]
            for key, exp in self.MB.expression.items():
                value = self.MB.process(bits, exp)
                if value is not None:
                    if isinstance(value, float):
                        new_row[key] = value
                    else:
                        for icol in range(value.shape[0]):
                            new_row[key + '_{}'.format(icol)] = value[icol]

            for key, exp in self.NI.expression.items():
                nnl, value = self.NI.process(bits, exp)
                if value is not None:
                    if isinstance(value, int) or isinstance(value, float):
                        new_row[key] = value
                        new_row['nnl'] = nnl
                else:
                    new_row[key] = 0

            if len(new_row) == len(values.columns):
                values = pd.concat(
                    [pd.DataFrame(new_row, columns=values.columns, index=[i]), values.loc[:]]).reset_index(
                    drop=True)

        # self._plot_dataframe_(values, 'time',
        #                       ['mobile_0', 'immobile_0', 'dissolved_2', 'mobile_0+immobile_0+dissolved_2'],
        #                       tag=fname)

        self._plot_dataframe_( values, 'time',
                              ['Iterations'],
                              tag=fname)



    def _plot_dataframe_(self, values, xkey, ykeys, tag=''):
        sec2hour = 60 * 60
        sec2day = sec2hour * 24
        sec2year = sec2day * 365
        kg2Mt = 1e3 * 1e6

        print('{}:\n\t\t time \t\t | \t\t TS \t\t | \t\t LI \t\t| \t\t NI \t\t | \t\t NI/TS \t\t| \t\t LI/NI \t\t |'.format(tag))
        for endtime in [25*sec2year, 50*sec2year, 750*sec2year, 1001*sec2year]:
            df = values[values['time']<endtime]
            print('\t -1000y -> {}y \t\t | \t {} \t | \t {} \t | \t {} \t| \t {} \t | \t {} \t |'.format(endtime/sec2year,
                                                                       df['dt'].count(),
                                                                       df['Iterations'].sum(),
                                                                       df['nnl'].sum(),
                                                                       df['nnl'].sum()/df['dt'].count(),
                                                                       df['Iterations'].sum()/df['nnl'].sum())
                  )

        for k in ykeys:
            yfield = self.process_keys(k, values)

            # plt.plot(values[xkey].to_numpy() / sec2day, (yfield.to_numpy() - yfield.to_numpy()[-1]) / kg2Mt, '-+',
            #          label=tag+'/'+k)
            plt.plot(values[xkey].to_numpy() / sec2year, yfield.to_numpy(), '-+',
                     label=tag + '/' + k)
            plt.xlim([-1000,750])
            # plt.plot(np.cumsum(values['dt'].to_numpy()) / sec2year, yfield.to_numpy(), '-+',
            #          label=tag + '/' + k)

        # plt.plot(values[xkey].to_numpy() / sec2year, np.min(values['dt']/sec2day,1e10/sec2day), label=tag + '/dt')

        # analytical for spe11b
        # q = 0.035
        # tn = np.linspace(0,50*sec2year,100)
        # tn = np.append(tn,np.linspace(50*sec2year, 1000*sec2year, 10) )
        # mn = []
        # for i in range(len(tn)-1):
        #     a = 0
        #     if tn[i] < 50*sec2year:
        #         a+=1
        #         if tn[i] > 25*sec2year:
        #             a+=1
        #     mn.append((tn[i+1]-tn[i])*a*q)
        # # plt.figure()
        # plt.plot(tn[1:]/ sec2day, np.cumsum(mn) / kg2Mt, '-ok', label='analytics')

    def process_keys(self, sin, v):
        if re.findall(r'\+', sin):
            bits = sin.split('+')
            bits = [item.strip() for item in bits]
            return self.process_keys(bits[0], v) + self.process_keys('+'.join(bits[1:]), v)
        elif re.findall(r'-', sin):
            bits = sin.split('-')
            bits = [item.strip() for item in bits]
            return self.process_keys(bits[0], v) - self.process_keys('+'.join(bits[1:]), v)
        elif re.findall(r'\*', sin):
            bits = sin.split('*')
            bits = [item.strip() for item in bits]
            return self.process_keys(bits[0], v) * self.process_keys('*'.join(bits[1:]), v)
        elif re.findall(r'/', sin):
            bits = sin.split('/')
            bits = [item.strip() for item in bits]
            return self.process_keys(bits[0], v) / self.process_keys('/'.join(bits[1:]), v)
        return v[sin]

    def _split_log_(self, fname):
        buffer = []
        time_tag = {}
        dt = {}
        buffer_ = ''
        with open(fname) as f:
            for line in f:
                if re.match(r'^Time', line):
                    buffer.append(buffer_)
                    time_tag[len(buffer)] = float(re.match(r'^Time: ((-)?\d+\.\d+e[+-]\d+)', line).group(1))
                    dt[len(buffer)] = float(re.findall(r'dt:(\s*(?:0|[1-9]\d*)(?:\.\d*)?)', line)[0])
                    buffer_ = ''
                elif re.match(r'Cleaning up events', line):
                    buffer.append(buffer_)
                else:
                    buffer_ += line
        header = buffer.pop(0)  # discard pre-simulation values
        return buffer, time_tag, dt, header


class Mass_balances:
    """
        Class design to handle mass info grep-ing and produce a DataFrame from it.
    """

    def __init__(self):

        self.expression = utils.expressions.mass_expression

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


class Newton_iterations:

    def __init__(self):
        self.expression = utils.expressions.newton_expression
        self.collabels = list(self.expression.keys())
        self.collabels.insert(0, 'time')
        self.collabels.insert(1, 'dt')
        self.collabels.insert(2, 'nnl')

    def process(self, buffer, exp):
        solvers = []
        t = re.findall(exp, buffer)
        if len(t) > 0:
            for matched in t:
                solvers.append(float(matched))
        else:
            return (0,None)
        return (len(t),np.sum(np.asarray(solvers), axis=0))
        # pass

    def _extract_time_(self, buffer):

        time = []
        for line in buffer.split('\n'):
            m = re.search(self.expression['dt'], line)
            n = re.search(self.expression['ndt'], line)
            o = re.search(self.expression['adt'], line)
            if m:
                time.append(float(m.group(1)))
            elif n:
                time.append(float(n.group(1)))
            elif o:
                time.append(float(o.group(1)))

        time.pop()
        arr = np.asarray(time)

        return arr

    def _extract_newit_(self, buffer):
        #
        newit = []
        for line in buffer.split('\n'):
            m = re.search(r'NewtonIter:(\s*\d{1,2})', line)
            if m:
                newit.append(int(m.group(1)))

        arr = np.asarray(newit)
        pos = np.where(np.diff(arr) < 0)

        return arr[pos]


if __name__ == "__main__":
    descr = 'Set of python script for post-processing GEOS\n'
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument('--log', metavar='logfile', nargs='+',
                        help='path to the logfile')

    args = parser.parse_args()

    if args.log:
        parser = LogParser()
        parser.process(fnames=args.log)
    else:
        parser.print_help()
