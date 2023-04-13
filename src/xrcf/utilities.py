
#//////////////////////////////////////////////////////////////////////////////
# convert channel to energy [eV]
#//////////////////////////////////////////////////////////////////////////////

def electronvolts(channel, α, β):
    """
    electronvolts(channel, α, β)

    Conversion of channel to energy in electron volts [eV].

    Parameters
    ----------
    channel : array_like of floats
        Channel(s).
    α : float
        Channel offset.
    β : float
        Conversion factor [channel / eV].

    Returns
    -------
    (channel - α) / β : array_like of floats

    """
    return (channel - α) / β

