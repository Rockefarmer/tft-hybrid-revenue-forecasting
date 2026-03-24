def train_model(model, train_df, val_df=None):
    model.fit(train_df, val_df)
    return model
