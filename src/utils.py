def annualize_return(daily_return):
    return (1 + daily_return.mean())**252 - 1
