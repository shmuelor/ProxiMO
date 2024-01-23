import numpy as np
from tool.utils import tensor2misc

class GeometricShapes(object):
    """
    Function class for generating masks of random geometric shapes
    """
    def __init__(self):
        pass
    
    @classmethod
    def create_random_cuboid_mask(cls, imgsz, ncub, ctmask=None, cubszlb=None, cubszub=None):
        """
        Create random cuboid mask given the number of cuboids.
        Inputs:
            imgsz: [x,y,z], size of whole image
            ncub: int, number of cuboids.
            ctmask: np.array, center mask. If given, only sample center from this region.
            cubszlb: int, lower bound of cuboid size in pixels. Default: 1/20 * imgsz 
            cubszub: int, upper bound of cuboid size in pixels. Default: 1/10 * imgsz
        Return:
            numpy array of size = imgsz
        """
        if ctmask is None:
            ctmask = np.ones(imgsz, dtype='bool')
        if cubszlb is None:
            cubszlb = max(1, round(1/20 * (np.prod(imgsz) ** (1/3))))
        if cubszub is None:
            cubszub = min(max(1, round(1/10 * (np.prod(imgsz) ** (1/3)))), min(imgsz))
        # logging.debug(cubszlb, cubszub)
        
        # sample cuboid centers
        ct_candidates = np.argwhere(ctmask==1)
        cubct = ct_candidates[np.random.choice(len(ct_candidates), ncub)]
        # sample cuboid sizes
        cubsz = np.array([np.random.randint(cubszlb, cubszub+1, ncub) for _ in range(3)]).T
        # generate and combine cuboid masks
        m = np.zeros(imgsz, dtype='uint8')
        for i in range(ncub):
            m = m | cls.create_cuboid_mask(imgsz, cubsz[i], cubct[i])
            
        return m
        
    @staticmethod
    def create_cuboid_mask(imgsz, cubsz, cubct):
        """
        Create one cuboid mask given image size, cuboid size and cuboid center.
        Inputs:
            imgsz: [x,y,z], size of whole image
            cubsz: [x,y,z], size of cuboid
            cubct: [x,y,z], center of cuboid (right of the two centers if size is even)
        Return:
            numpy array of size = imgsz
        """
        m = np.zeros(imgsz, dtype='uint8')
        m[max(cubct[0]-cubsz[0]//2, 0) : min(cubct[0]-cubsz[0]//2+cubsz[0], imgsz[0]), \
          max(cubct[1]-cubsz[1]//2, 0) : min(cubct[1]-cubsz[1]//2+cubsz[1], imgsz[1]), \
          max(cubct[2]-cubsz[2]//2, 0) : min(cubct[2]-cubsz[2]//2+cubsz[2], imgsz[2])  \
         ] = 1
        return m
        

class IsotropicLesionAugmentation(object):
    def __init__(self):
        pass
    
    @staticmethod
    def add_isotropic_cuboid(sti, brain_mask, ncub, cubszlb=None, cubszub=None, returnM=False):
        """
        Add isotropic cuboid lesions to STI.
        Inputs:
            sti: [x,y,z,6], original sti
            brainmask: [x,y,z], binary brain mask
            ncub: int, number of cuboids
            cubszlb: int, lower bound of cuboid size in pixels. Default: 1/20 * (xyz)^(1/3)
            cubszub: int, upper bound of cuboid size in pixels. Default: 1/10 * (xyz)^(1/3)
            returnM: bool, if True, return isotropic region mask.
        Return:
            np.array, new sti
        """
        # get mms, msa, pev
        L, V, avg, ani, V1 = tensor2misc(sti)
        mms = avg
        msa = ani
        
        # cuboid mask
        imgsz = sti.shape[:3]
        ma = GeometricShapes.create_random_cuboid_mask(
            imgsz, ncub, ctmask=brain_mask, cubszlb=cubszlb, cubszub=cubszub)
        
        sti_aug = sti.copy()
        sti_aug[ma==1,0] = mms[ma==1]
        sti_aug[ma==1,3] = mms[ma==1]
        sti_aug[ma==1,5] = mms[ma==1]
        sti_aug[ma==1,1] = 0
        sti_aug[ma==1,2] = 0
        sti_aug[ma==1,4] = 0
        
        sti_aug = sti_aug * brain_mask[:,:,:,None]
        if returnM:
            return sti_aug, ma
        else:
            return sti_aug
    