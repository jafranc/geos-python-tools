import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import Data, Conversion, logging


def _integrate_3_(pts, fields, box):
    """ Simplistic integration equivalent to Paraview integrate variables """
    xmin, ymin, zmin = box[0]
    xmax, ymax, zmax = box[1]
    ii = np.intersect1d(
        np.intersect1d(np.intersect1d(np.argwhere((pts[:, 0] > xmin)), np.argwhere((pts[:, 0] < xmax))),
                       np.intersect1d(np.argwhere((pts[:, 2] > zmin)), np.argwhere((pts[:, 2] < zmax)))),
        np.intersect1d(np.argwhere((pts[:, 1] > ymin)), np.argwhere((pts[:, 1] < ymax)))
    )
    return fields[ii].sum()


def _integrate_gradient_2_(fn, vol, box, dims):
    """ Integrate gradient of fields re-interpolate on regular grid to deal with arbitrary mesh gradient """
    xmin, ymin, zmin = box[0]
    xmax, ymax, zmax = box[1]
    x, z = np.meshgrid(np.linspace(xmin, xmax, dims[0]), np.linspace(zmin, zmax, dims[1]))
    dx, dy, dz = ((xmax - xmin) / dims[0], (ymax - ymin) / 1, (zmax - zmin) / dims[1])
    res = fn(np.asarray([x.flatten(), z.flatten()]).transpose())
    res /= (vol(np.asarray([x.flatten(), z.flatten()]).transpose()) + 1e-12)
    dres = np.gradient(np.reshape(res, (dims[0], dims[1])))
    return (np.sqrt(np.power(dres[0], 2) + np.power(dres[1], 2)) * dx * dy * dz).sum()


def _integrate_2_(pts, fields, box):
    """ Simplistic integration equivalent to Paraview integrate variables """
    xmin, ymin, zmin = box[0]
    xmax, ymax, zmax = box[1]
    ii = np.intersect1d(np.intersect1d(np.argwhere((pts[:, 0] > xmin)), np.argwhere((pts[:, 0] < xmax))),
                        np.intersect1d(np.argwhere((pts[:, 2] > zmin)), np.argwhere((pts[:, 2] < zmax))))
    return fields[ii].sum()


def _integrate_gradient_3_(fn, vol, box, dims):
    """ Integrate gradient of fields re-interpolate on regular grid to deal with arbitrary mesh gradient """
    xmin, ymin, zmin = box[0]
    xmax, ymax, zmax = box[1]
    x, y, z = np.meshgrid(np.linspace(xmin, xmax, dims[0]), np.linspace(ymin, ymax, dims[1]),
                          np.linspace(zmin, zmax, dims[2]))
    dx, dy, dz = ((xmax - xmin) / dims[0], (ymax - ymin) / dims[1], (zmax - zmin) / dims[2])
    res = fn(np.asarray([x.flatten(), y.flatten(), z.flatten()]).transpose())
    res /= (vol(np.asarray([x.flatten(), y.flatten(), z.flatten()]).transpose()) + 1e-12)
    dres = np.gradient(np.reshape(res, (dims[0], dims[1], dims[2])))
    return (np.sqrt(np.power(dres[0], 2) + np.power(dres[1], 2) + np.power(dres[2], 2)) * dx * dy * dz).sum()


class Sparse_Data(Data):
    """ Class for handling from vtm time series to sparse data SPE11-CSP"""

    def __init__(self, version, solubility_file, units):
        super().__init__(version)

        self.converters = [('sec', 1), ('kg', 1), ('Pa', 1)]

        if 's' in units:
            self.converters[0] = ('sec', Conversion.SEC2SEC)
        elif 'h' in units:
            self.converters[0] = ('hour', Conversion.SEC2HOUR)
        elif 'd' in units:
            self.converters[0] = ('day', Conversion.SEC2DAY)
        elif 'y' in units:
            self.converters[0] = ('year', Conversion.SEC2YEAR)

        if 'kg' in units:
            self.converters[1] = ('kg', Conversion.KG2KG)
        elif 't' in units:
            self.converters[1] = ('t', Conversion.KG2T)
        elif 'g' in units:
            self.converters[1] = ('g', Conversion.KG2G)

        if 'Pa' in units:
            self.converters[2] = ('Pa', Conversion.PA2PA)
        elif 'bar' in units:
            self.converters[2] = ('bar', Conversion.PA2BAR)

        self.path_to_solubility = solubility_file

        # geom in meters
        self.boxes = {'Whole': [(0.0, 0.0, -1.2), (2.8, .01, 0.0)]}
        self.PO1 = [1.5, -0.7]
        self.PO2 = [1.7, -0.1]
        self.schedule = np.arange(0, 5 * Conversion.SEC2DAY, 30000.)
        offx, offz = (0., 0.)

        if self.version == 'b':
            self.boxes['Whole'] = [(item[0] * 3000, 1. / 0.01 * item[1], item[2] * 1000) for item in
                                   self.boxes['Whole']]
            self.PO1 = [self.PO1[0] * 3000, self.PO1[1] * 1000]
            self.PO2 = [self.PO2[0] * 3000, self.PO2[1] * 1000]
            self.schedule = np.arange(0., 1000 * Conversion.SEC2YEAR, 10 * Conversion.SEC2TENTHOFYEAR)
        elif self.version == 'c':
            self.schedule = np.arange(0., 1000 * Conversion.SEC2YEAR, 10 * Conversion.SEC2TENTHOFYEAR)
            # self.schedule = self.schedule[::10]
            self.boxes['Whole'] = [(item[0] * 3000, 5000. / 0.01 * item[1], item[2] * 1000) for item in
                                   self.boxes['Whole']]
            self.boxes['Whole'][0] = (0., 0., 0.)
            self.PO1 = [self.PO1[0] * 3000, 2500, (self.PO1[1] + 1.2) * 1000]
            self.PO2 = [self.PO2[0] * 3000, 2500, (self.PO2[1] + 1.2) * 1000]
            offx, offz = (100., 150.)

        # origin
        Ox, Oy, Oz = self.boxes['Whole'][0]
        # full length
        Lx, Ly, Lz = self.boxes['Whole'][1]
        Lx -= Ox
        Ly -= Oy
        Lz -= Oz

        self.boxes['A'] = [(Ox + 1.1 / 2.8 * Lx, Oy, Oz), (Ox + Lx - offx, Oy + Ly, Oz + 0.6 / 1.2 * Lz + offz)]
        self.boxes['B'] = [(Ox + offx, Oy, Oz + 0.6 / 1.2 * Lz + offz), (Ox + 1.1 / 2.8 * Lx, Oy + Ly, Oz + Lz + offz)]
        self.boxes['C'] = [(Ox + 1.1 / 2.8 * Lx, Oy, Oz + 0.1 / 1.2 * Lz + offz),
                           (Ox + 2.6 / 2.8 * Lx, Oy + Ly, Oz + 0.5 / 1.2 * Lz + offz)]

    def process(self, directory, ifile):

        super().process(directory, ifile)

        self._write_(directory, ifile)
        self._plot_(directory)

    #
    def _thread_this_(self, ifile,directory ,olist_ , ff, time):

        pts_from_vtk, fields = self._process_time_(ifile, time, olist=olist_)

        # some lines for MC magic number
        if self.version[0] == 'a' or self.version[0] == 'c':
            fields['mCO2Max'] = ff(fields['pres'], 293) * fields['rL']
        else:
            # convert it to kgCO2/m3Brine
            fields['mCO2Max'] = ff(fields['pres'], fields['temp']) * fields['rL']
        self.formula['M_C'] = 'mCO2/mCO2Max'

        for key, form in self.formula.items():
            fields[key] = self.process_keys(form, fields)

        logging.info(f'Interpolating reported fields')
        if self.version[0] != "c":
            fn = self._get_interpolate_(pts_from_vtk,
                                        {'pres': fields['pres'], 'M_C': fields['M_C'], 'vol': fields['vol']})
        else:
            fn = self._get_interpolate_(pts_from_vtk,
                                        {'pres': fields['pres'], 'M_C': fields['M_C'], 'vol': fields['vol']},
                                        nskip=1)

        line = [time]
        # deal with P1 and P2
        logging.info(f'Writing sparse time {time:2}')
        p1 = fn['pres'](self.PO1)
        p2 = fn['pres'](self.PO2)
        line.extend([p1[0], p2[0]])
        # deal box A & B
        logging.info(f'Integrating for boxes A and B')
        if self.version[0] != "c":
            for box_name, box in self.boxes.items():
                if box_name in ['A', 'B']:
                    line.extend([
                        _integrate_2_(pts_from_vtk, fields['mMobile'], box),
                        _integrate_2_(pts_from_vtk, fields['mImmobile'], box),
                        _integrate_2_(pts_from_vtk, fields['mDissolved'], box),
                        _integrate_2_(pts_from_vtk, fields['mSeal'], box)
                    ])
            # #deal box C
            logging.info(f'Integrating for box C')
            line.append(
                _integrate_gradient_2_(fn['M_C'], fn['vol'], self.boxes['C'], (150, 50)))
            # #deal sealTot
            line.append(
                _integrate_2_(pts_from_vtk, fields['mSeal'], self.boxes['Whole']))
        else:
            for box_name, box in self.boxes.items():
                if box_name in ['A', 'B']:
                    line.extend([
                        _integrate_3_(pts_from_vtk, fields['mMobile'], box),
                        _integrate_3_(pts_from_vtk, fields['mImmobile'], box),
                        _integrate_3_(pts_from_vtk, fields['mDissolved'], box),
                        _integrate_3_(pts_from_vtk, fields['mSeal'], box)
                    ])
                # #deal box C
            logging.info(f'Integrating for box C')
            line.append(
                _integrate_gradient_3_(fn['M_C'], fn['vol'], self.boxes['C'], (150, 10, 50)))
            # #deal sealTot
            line.append(_integrate_3_(pts_from_vtk, fields['mSeal'], self.boxes['Whole']))

            df = pd.DataFrame(data=[line],
                            columns=['t[s]', 'p1[Pa]', 'p2[Pa]', 'mobA[kg]', 'immA[kg]', 'dissA[kg]', 'sealA[kg]',
                                     'mobB[kg]', 'immB[kg]', 'dissB[kg]', 'sealB[kg]', 'M_C[m]', 'sealTot[kg]'])
            #intermediate write
            fname = './' + directory + f'/{time:2}_spe11' + self.version + '_time_series.csv'
            df.to_csv(fname, index=False, float_format='%.3f')

            return df

    def _write_(self, directory, ifile):
        # note that solubility is in KgCO2/KgBrine
        ff = self._process_solubility_(self.path_to_solubility)

        # preprocess input list from desired output
        import re
        olist_ = set(
            [item for name in self.name_indirection.keys() for item in re.findall(r'(\w+)_\d|(\w+)', name)[0] if
             len(item) > 0])

        # for time in tqdm(self.schedule):
        from tqdm.contrib.concurrent import process_map
        # import multiprocessing as mp

        from functools import partial
        pdlist = []
        # pool = mp.Pool(processes=8)
        pdlist.append(process_map(partial(self._thread_this_, ifile, directory, olist_, ff), self.schedule, max_workers=16))
        # pool.close()
        # pool.join()

        # write off panda dataframe ordered by time
        df = pd.concat(pdlist[0], ignore_index=True)
        df.sort_values(by=['t[s]'])
        df.to_csv('./' + directory + '/spe11' + self.version + '_time_series.csv', float_format='%.3f', index=False)

    # def _integrate_gradient_2_(self, fn, vol, box, dims):
    #     """ Integrate gradient of fields re-interpolate on regular grid to deal with arbitrary mesh gradient """
    #     xmin, ymin, zmin = box[0]
    #     xmax, ymax, zmax = box[1]
    #     x, z = np.meshgrid(np.linspace(xmin, xmax, dims[0]), np.linspace(zmin, zmax, dims[1]))
    #     dx, dy, dz = ((xmax - xmin) / dims[0], (ymax - ymin) / 1, (zmax - zmin) / dims[1])
    #     res = fn(np.asarray([x.flatten(), z.flatten()]).transpose())
    #     res /= (vol(np.asarray([x.flatten(), z.flatten()]).transpose()) + 1e-12)
    #     dres = np.gradient(np.reshape(res, (dims[0], dims[1])))
    #     return (np.sqrt(np.power(dres[0], 2) + np.power(dres[1], 2)) * dx * dy * dz).sum()
    #
    # def _integrate_gradient_3_(self, fn, vol, box, dims):
    #     """ Integrate gradient of fields re-interpolate on regular grid to deal with arbitrary mesh gradient """
    #     xmin, ymin, zmin = box[0]
    #     xmax, ymax, zmax = box[1]
    #     x, y, z = np.meshgrid(np.linspace(xmin, xmax, dims[0]), np.linspace(zmin, zmax, dims[2]))
    #     dx, dy, dz = ((xmax - xmin) / dims[0], (ymax - ymin) / dims[1], (zmax - zmin) / dims[2])
    #     res = fn(np.asarray([x.flatten(), y.flatten(), z.flatten()]).transpose())
    #     res /= (vol(np.asarray([x.flatten(), y.flatten(), z.flatten()]).transpose()) + 1e-12)
    #     dres = np.gradient(np.reshape(res, (dims[0], dims[1])))
    #     return (np.sqrt(np.power(dres[0], 2) + np.power(dres[1], 2)) * dx * dy * dz).sum()
    def _plot_(self, directory):

        import pandas as pd
        df = pd.read_csv(directory + '/spe11' + self.version + '_time_series.csv')
        fig, axs = plt.subplots(2, 2)
        (time_name, time_unit), (mass_name, mass_unit), (pressure_name, pressure_unit) = self.converters
        # pressures
        axs[0][0].plot(df['t[s]'].to_numpy() / time_unit, df['p1[Pa]'].to_numpy() / pressure_unit,
                       label=f'pressure 1 [{pressure_name}]')
        axs[0][0].plot(df['t[s]'].to_numpy() / time_unit, df['p2[Pa]'].to_numpy() / pressure_unit,
                       label=f'pressure 2 [{pressure_name}]')
        axs[0][0].legend()
        # box A
        axs[0][1].plot(df['t[s]'].to_numpy() / time_unit, df['mobA[kg]'].to_numpy() / mass_unit,
                       label=f'mobile CO2 [{mass_name}]')
        axs[0][1].plot(df['t[s]'].to_numpy() / time_unit, df['immA[kg]'].to_numpy() / mass_unit,
                       label=f'immobile CO2 [{mass_name}]')
        axs[0][1].plot(df['t[s]'].to_numpy() / time_unit, df['dissA[kg]'].to_numpy() / mass_unit,
                       label=f'dissolved CO2 [{mass_name}]')
        axs[0][1].plot(df['t[s]'].to_numpy() / time_unit, df['sealA[kg]'].to_numpy() / mass_unit,
                       label=f'seal CO2 [{mass_name}]')
        axs[0][1].legend()
        axs[0][1].set_title('boxA')
        # box B
        axs[1][0].plot(df['t[s]'].to_numpy() / time_unit, df['mobB[kg]'].to_numpy() / mass_unit,
                       label=f'mobile CO2 [{mass_name}]')
        axs[1][0].plot(df['t[s]'].to_numpy() / time_unit, df['immB[kg]'].to_numpy() / mass_unit,
                       label=f'immobile CO2 [{mass_name}]')
        axs[1][0].plot(df['t[s]'].to_numpy() / time_unit, df['dissB[kg]'].to_numpy() / mass_unit,
                       label=f'dissolved CO2 [{mass_name}]')
        axs[1][0].plot(df['t[s]'].to_numpy() / time_unit, df['sealB[kg]'].to_numpy() / mass_unit,
                       label=f'seal CO2 [{mass_name}]')
        axs[1][0].legend()
        axs[1][0].set_title('boxB')
        # boxC
        axs[1][1].plot(df['t[s]'].to_numpy() / time_unit, df['M_C[m]'].to_numpy(), label='M_C[m]')
        axs[1][1].legend()
        axs[1][1].set_title('boxC')

        fig.tight_layout()
        fig.savefig('./' + directory + '/spe11' + self.version + '_timeseries.png', bbox_inches='tight')
