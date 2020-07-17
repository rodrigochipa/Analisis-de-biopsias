import torch

import sys
sys.path.append("../NucleiSegmentation/")
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model

class cGAN_nuclei():
    
    def __init__(self, gpu_ids=[]):
        opt = TestOptions()#.parse()
        opt.nThreads = 1   # test code only supports nThreads = 1
        opt.batchSize = 1  # test code only supports batchSize = 1
        opt.serial_batches = True  # no shuffle
        opt.no_flip = True  # no flip
        opt.display_id = -1  # no visdom display
        opt.dataset_mode = "single"
        opt.dataroot = "."
        opt.phase = "test"
        opt.loadSize = 256
        opt.fineSize = 256
        opt.isTrain = False
        opt.input_nc = 3
        opt.output_nc = 3
        opt.gpu_ids = gpu_ids
        opt.name = "NU_SEG"
        opt.model_suffix = ""
        opt.checkpoints_dir = "../NucleiSegmentation/checkpoints/"
        opt.model = "test"
        opt.ngf = 64
        opt.norm = "instance"
        opt.which_model_netG = "unet_256"
        opt.resize_or_crop = "resize_and_crop"
        opt.which_epoch = "latest"
        opt.no_dropout = "store_true"
        opt.init_type = "normal"
        opt.init_gain = 0.02
        opt.verbose = ""
        opt.which_direction = "BtoA"
        data_loader = CreateDataLoader(opt)
        dataset = data_loader.load_data()
        model = create_model(opt)
        model.setup(opt)

        self.model = model

        
    def get_contours(self, numpy_image):
        # TODO: CHECK https://github.com/abhishekvahadane/CodeRelease_ColorNormalization
        # for normalization used in original work
        
        assert numpy_image.shape == (256, 256, 3), "Image is not 256x256x3"
                                       
        with torch.no_grad():
            torch_image = torch.from_numpy(numpy_image.T.astype('float32'))/255.
            data = {'A': torch_image.unsqueeze(0), 'A_paths': '.'}
            self.model.set_input(data)
            self.model.test()
            visuals = self.model.get_current_visuals()
            contours = visuals['fake_B'].squeeze(0).cpu().numpy().T
            return (0.5*(contours + 1.)*255).astype('uint8')
        
