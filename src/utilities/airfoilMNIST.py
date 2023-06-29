def prefilter_dataset(vtu_list, *, aoa_limits, mach_limits):
    def get_idx(x, *, name):
        if name == 'aoa':
            return float(x.rsplit('.', 1)[0].split('_')[1])
        if name == 'mach':
            return float(x.rsplit('.', 1)[0].split('_')[2])

    if isinstance(mach_limits, tuple) and isinstance(aoa_limits, tuple):
        lst = [x for x in vtu_list if
               (mach_limits[0] <= get_idx(x, name='mach') <= mach_limits[1])]
        lst = [x for x in lst if
               (aoa_limits[0] <= get_idx(x, name='aoa') <= aoa_limits[1])]

        return lst
