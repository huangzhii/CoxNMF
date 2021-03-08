
def expression_filter(x, meanq = 0.2, varq = 0.2):
    '''
    Parameters
    ----------
    x     : Real-valued expression matrix with rownames indicating
            gene ID or gene symbol.
    meanq : By which genes with low expression mean across samples are filtered out.
    varq  : By which genes with low expression variance across samples are filtered out.
        
    Returns
    -------
    x     : Real-valued expression matrix with rownames indicating
            gene ID or gene symbol.
    '''
    mean_quantile = x.mean(axis = 1).quantile(q=meanq)
    x = x.loc[x.mean(axis = 1) >= mean_quantile,:]
    var_quantile = x.var(axis = 1).quantile(q=varq)
    x = x.loc[x.var(axis = 1) >= var_quantile,:]
    return x