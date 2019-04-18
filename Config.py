class Config():
    """
    The Configration of the FaceNet
    """

    """
    embedding_size       : The output of the size of feature.
    num_classes          : The number of classes of people.
    margin               : The margin of triplet loss.
    
    """
    embedding_size = 128
    num_classes = 54
    margin = 0.5

    """
    batch_size           : The batch of training.
    num_workers          : The workers to generate data.
    start_epoch          : The epoch to start with. If not 0, it will continue the training.
    num_epochs           : The total epoch to train.
    """
    batch_size = 64
    num_workers = 1
    start_epoch = 0
    num_epochs = 100

    """
    base_learning_rate   : The base learning rate. The learning rate will start after the "warm_up_epoch" 
                           if use_warmup is True, else it will work at the beginning.
    start_learning_rate  : The start learning rate. The learning rate of the warm up start learning rate.
                           IT ONLY WORKS WHEN THE use_warmup TRUE.
    warmup_epoch         : The epoch to use warm up learning rate. IT ONLY WORKS WHEN THE use_warmup TRUE.
    use_warmup           : To use warm up or not.
    """
    base_learning_rate = 0.01
    start_learning_rate = 1e-5
    warmup_epoch = 5
    use_warmup = False

    """
    image_size           : The size of input image (image_size,image_size).
    del_classifier       : The flag to delete the classifier layers. Set it True when change your data classes.
    triplet_lambuda      : The lambuda forword the triplet loss. The triplet loss is (triplet*triplet_lambuda).
    """
    image_size = 56

    del_classifier = False
    triplet_lambuda = 5