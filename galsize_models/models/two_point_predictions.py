"""
"""
from halotools.mock_observables import return_xyz_formatted_array


__all__ = ('clustering_sample_iterator', )


def clustering_sample_iterator(table, *masks, **kwargs):
    """
    Examples
    --------
    >>> from astropy.table import Table
    >>> t = Table()
    >>> t['halo_x'] = [1, 2, 3, 4]
    >>> t['halo_y'] = [1, 2, 3, 4]
    >>> t['halo_z'] = [1, 2, 3, 4]
    >>> mask1 = t['halo_x'] > 2
    >>> mask2 = t['halo_y'] <= 3
    >>> subtables = list(clustering_sample_iterator(t, mask1, mask2, zspace=False))
    """
    zspace = kwargs.get('zspace', True)
    period = kwargs.get('period', None)

    if (period is None) and (zspace is True):
        raise ValueError("Must specify ``period`` keyword argument if ``zspace`` is True\n"
                "Use ``np.inf`` to ignore periodic boundary conditions")

    for i, mask in enumerate(masks):
        sample = table[mask]

        if zspace:
            sample_pos = return_xyz_formatted_array(sample['halo_x'],
                    sample['halo_y'], sample['halo_z'],
                    velocity=sample['halo_vz'], velocity_distortion_dimension='z', period=period)
            msg = ("Input ``table`` must have the following columns: \n"
                "``halo_x``, ``halo_y``, ``halo_z``, ``halo_vz``")
            assert set(('halo_x', 'halo_y', 'halo_z', 'halo_vz')) <= set(list(table.keys())), msg
        else:
            sample_pos = return_xyz_formatted_array(sample['halo_x'],
                    sample['halo_y'], sample['halo_z'])
            msg = ("Input ``table`` must have the following columns: \n"
                "``halo_x``, ``halo_y``, ``halo_z``")
            assert set(('halo_x', 'halo_y', 'halo_z')) <= set(list(table.keys())), msg

        yield sample_pos
