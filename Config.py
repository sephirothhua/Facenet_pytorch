class Config():
    embedding_size = 128
    num_classes = 54

    batch_size = 64
    num_workers = 1
    margin = 0.5

    start_epoch = 85
    num_epochs = 100

    base_learning_rate = 0.01
    start_learning_rate = 1e-5
    warmup_epoch = 5
    use_warmup = True

    image_size = 56

    del_classifier = True