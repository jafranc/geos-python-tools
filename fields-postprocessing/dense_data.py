import matplotlib.pyplot as plt
import numpy as np

from data import Data, Conversion, logging


class Dense_Data(Data):
    """ Class for handling from vtm time series to dense data SPE11-CSP """

    def __init__(self, version, solubility_file):

        super().__init__(version)
        self.path_to_solubility = solubility_file
        self.phydims = (2.8, 1., 1.2)
        self.dims = (280, 1, 120)
        self.offset = [0., 0., -1.2]
        self.filename_converter, self.filename_marker = (Conversion.SEC2HOUR, 'h')

        if version[0] == 'b':
            self.phydims = (2.8 * 3000, 1., 1.2 * 1000)
            self.dims = (840, 1, 120)
            self.offset = [0., 0., -1200.]
            #
            self.schedule = np.arange(0., 1000 * Conversion.SEC2YEAR, 50 * Conversion.SEC2TENTHOFYEAR)
            self.filename_converter, self.filename_marker = (Conversion.SEC2YEAR, 'y')
        elif version[0] == 'c':
            self.phydims = (2.8 * 3000, 5000., 1.2 * 1000 + 150)
            self.dims = (168, 100, 120)
            self.offset = [0., 0., 0.]
            #
            self.schedule = list(np.arange(0, 50 * Conversion.SEC2YEAR, 5 * Conversion.SEC2YEAR))
            self.schedule.extend([75 * Conversion.SEC2YEAR, 100 * Conversion.SEC2YEAR])
            self.schedule.extend(
                np.arange(100 * Conversion.SEC2YEAR, 500 * Conversion.SEC2YEAR, 50 * Conversion.SEC2YEAR))
            self.schedule.extend(
                np.arange(500 * Conversion.SEC2YEAR, 1000 * Conversion.SEC2YEAR, 100 * Conversion.SEC2YEAR))
            self.filename_converter, self.filename_marker = (Conversion.SEC2YEAR, 'y')

        Nx, Ny, Nz = self.dims
        Lx, Ly, Lz = self.phydims
        self.x, self.y, self.z = np.meshgrid(np.linspace(Lx / 2 / Nx, (2 * Nx - 1) * Lx / 2 / Nx, Nx) + self.offset[0],
                                             np.linspace(Ly / 2 / Ny, (2 * Ny - 1) * Ly / 2 / Ny, Ny) + self.offset[1],
                                             np.linspace(Lz / 2 / Nz, (2 * Nz - 1) * Lz / 2 / Nz, Nz) + self.offset[2],
                                             indexing='xy')

    def _thread_this_(self, directory, ifile, ff, olist_, itime):
        time = self.schedule[itime]
        pts_from_vtk, fields = self._process_time_(ifile, time, olist=olist_)
        # some lines for MC magic number
        if self.version[0] == 'a':  # TODO change it when comes to thermal results
            fields['mCO2Max'] = ff(fields['pres'], 293) * fields['rL']
        else:
            # convert it to kgCO2/m3Brine
            fields['mCO2Max'] = ff(fields['pres'], fields['temp']) * fields['rL']
        self.formula['M_C'] = 'mCO2/mCO2Max'

        for key, form in self.formula.items():
            fields[key] = self.process_keys(form, fields)

        fn = self._get_interpolate_(pts_from_vtk, fields)
        self._write_(time, fn, directory)

        return #(time, fn['satg'], fn['mCO2'], fn['temp'])

    def process(self, directory, ifile):

        super().process(directory, ifile)
        ff = self._process_solubility_(self.path_to_solubility)

        # preprocess input list from desired output
        import re
        olist_ = set(
            [item for name in self.name_indirection.keys() for item in re.findall(r'(\w+)_\d|(\w+)', name)[0] if
             len(item) > 0])

        # import multiprocessing as mp
        from tqdm.contrib.concurrent import process_map
        from functools import partial

        logging.info(f'Pooling the work')
        fn_stack = []

        # pool = mp.Pool(processes=4)
        # fn_stack.append(
        process_map(partial(self._thread_this_, directory, ifile, ff, olist_), range(len(self.schedule), max_workers=12))
        # pool.close()
        # pool.join()
        # func = partial(self._thread_this_, directory, ifile, ff, olist_)
        # for itime in range(len(self.schedule)):
        #     func(itime)

        # # to plot as would be printed
        # baseFileName = directory + '/plot'
        # import matplotlib.pyplot as plt
        # fig = {'satg': plt.figure(figsize=(18, 6)),
        #        'mCO2': plt.figure(figsize=(18, 6))}
        #
        # if 'temp' in self.name_indirection.values():
        #     fig['temp'] = plt.figure(figsize=(18, 6))
        #
        # csv_keys_translation = {'x': '# #x[m]', 'y': ' y[m]',
        #                         'z': ' z[m]',
        #                         'satg': ' gas saturation[-]',
        #                         'mCO2': ' mass fraction of CO2 in liquid[-]',
        #                         'temp': ' temperature[C]'}
        #
        # # for itime, time in enumerate(self.schedule[::int(len(self.schedule) / 10)][1:]):
        # for itime, time in enumerate(self.schedule):
        #     import pandas as pd
        #     fname = './' + directory + '/spatial_map_' + "{time:2}".format(
        #         time=time / self.filename_converter) + self.filename_marker + '.csv'
        #     data = pd.read_csv(fname)
        #     data = data.drop(0)  # miss write from numpy
        #     if self.version[0] != "c":
        #         # as for out regular grid size it is faster just to dump the values
        #         fn = lambda k: data[csv_keys_translation[k]]
        #     else:
        #         pass
        #         # get_index = lambda time_ : [index for index in range(len(fn_stack[0])) if fn_stack[0][index][0] == time_][0]
        #         # fn = self.get_interpolate_(zip(data[csv_keys_translation['x'],data[csv_keys_translation['y']],data[csv_keys_translation['z']]),
        #         #                            {''})
        #         # for i,key in enumerate(fig.keys()):
        #         #     fn[key] = fn_stack[0][get_index(time)][i+1]
        #
        #     # self._plot_((itime, time), fig, fn, self.dims)
        #
        # for key, _ in fig.items():
        #     fig[key].savefig(f'{baseFileName}_{key}.png', bbox_inches='tight')

    def _write_(self, time, fn, directory):

        logging.info(f'Writing spatial_map_{time:2}')
        file_name = './' + directory + '/spatial_map_' + "{time:2}".format(
            time=time / self.filename_converter) + self.filename_marker + '.csv'
        file_header = ("#x[m], y[m], z[m], pressure[Pa], gas saturation[-], mass fraction of CO2 in liquid[-], "
                       "mass fraction of H2O in vapor[-], phase mass density gas[kg/m3], phase mass density water [kg/m3], total mass CO2[kg]\n")
        file_fmt = "%.3e, %.3e, %.3e, %.3e, %.3e, %.3e, %.3e, %.3e, %.3e, %.3e"
        if self.version[0] != "c":
            xyz = np.asarray([self.x.flatten(), self.z.flatten()]).transpose()
        else:
            xyz = np.asarray([self.x.flatten(), self.y.flatten(), self.z.flatten()]).transpose()

        output = np.asarray([self.x.flatten(), self.y.flatten(), self.z.flatten(),
                             fn['pres'](xyz), fn['satg'](xyz), fn['mCO2'](xyz), fn['mH2O'](xyz),
                             fn['rG'](xyz), fn['rL'](xyz), fn['mTotal'](xyz)])

        if 'temp' in self.name_indirection.values():
            file_header = file_header[:-1] + ", temperature[C]\n"
            file_fmt += ", %.3e"
            output = np.vstack([output, fn['temp'](xyz) - Conversion.K2C])

        np.savetxt(file_name, output.transpose(), fmt=file_fmt, delimiter=',', header=file_header)

    def _plot_(self, ttime, figdict, fn, dims):
        """ To check good rendition of plots - extract of B.Flemish code"""
        nx, ny, nz = dims
        Lx, Ly, Lz = self.phydims
        itime, time = ttime

        if ny > 1:
            import matplotlib.gridspec as gridspec
            xgrid, zgrid = np.meshgrid(np.linspace(0 + self.offset[0], Lx + self.offset[0], nx + 1),
                                       np.linspace(0 + self.offset[1], Lz + self.offset[1], nz + 1), indexing='xy')
            ygrid, zgrid1 = np.meshgrid(np.linspace(0 + self.offset[0], Ly + self.offset[0], ny + 1),
                                        np.linspace(0 + self.offset[1], Lz + self.offset[1], nz + 1), indexing='xy')
            # ygrid, xgrid, zgrid = np.meshgrid(
            #     np.linspace(0 + self.offset[0], Ly + self.offset[0], ny + 1),
            #     np.linspace(0 + self.offset[0], Lx + self.offset[0], nx + 1),
            #     np.linspace(0 + self.offset[1], Lz + self.offset[1], nz + 1),
            #     indexing='xy')
        else:
            xgrid, zgrid = np.meshgrid(np.linspace(0 + self.offset[0], Lx + self.offset[0], nx + 1),
                                       np.linspace(0 + self.offset[1], Lz + self.offset[1], nz + 1), indexing='xy')

        for key, fig in figdict.items():
            if ny == 1:
                assert(isinstance(fn,list))
                ax = fig.add_subplot(3, 3, itime + 1)
                im = ax.pcolormesh(xgrid, zgrid,
                                   np.reshape(fn(key).to_numpy(), (nx, nz)).transpose(),
                                   shading='flat',
                                   cmap='coolwarm')
                ax.axis([self.x.min(), self.x.max(), self.z.min(), self.z.max()])
                ax.axis('scaled')
                # fig.colorbar(im, orientation='horizontal')
                ax.set_title(
                    '{:s} at t = {:1.0f} {:s}'.format(key, time / self.filename_converter, self.filename_marker))
            else:
                assert(isinstance(fn,dict))
                outer = gridspec.GridSpec(2, 5)
                inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[itime])
                xyz = np.asarray([self.x.flatten(), self.y.flatten(), self.z.flatten()]).transpose()
                ax = plt.Subplot(fig, inner[0])
                _ = ax.pcolormesh(xgrid,
                                  zgrid,
                                  np.reshape(fn[key](xyz),(nx,ny,nz)).transpose((1,0,2))[int(ny/2),:,:].transpose(),
                                  shading='flat',
                                  cmap='coolwarm')
                ax.axis('scaled')
                ax.set_title(
                    '{:s} at t = {:1.0f} {:s}'.format(key, time / self.filename_converter, self.filename_marker))
                fig.add_subplot(ax)
                #
                ax = plt.Subplot(fig, inner[1])
                im = ax.pcolormesh(ygrid,
                                   zgrid1,
                                   np.reshape(fn[key](xyz),(nx,ny,nz))[int(nx/2),:,:].transpose(),
                                   shading='flat',
                                   cmap='coolwarm')
                ax.axis('scaled')
                fig.add_subplot(ax)
                # fig.colorbar(im, orientation='horizontal')
