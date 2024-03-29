class d():
    def __init__(self):
        self.img_height = 64
        self.img_width = 64
        self.noise_dim = 100
        self.emb_dim = 1024
        self.projected_em_dim = 128
        self.batch_size = 429
        self.learning_rate = 0.0002
        self.epochs = 200
        self.cuda = True
        self.data_dir = 'birds.hdf5'
        self.save_dir = 'saved_models'
