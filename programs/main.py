from Model.Image import Image
from View.Painter import Painter
from Controller.Masker import Masker
from Controller.MaskMover import MaskMover
from Controller.PoissonBlender import PoissonBlender
import cv2, sys
#cv2.IMREAD_GRAYSCALE
def main(args):
    source = Image(args[1])
    target = Image(args[2])
    # './data/raw/sig_wood/src.jpg'
    # './data/raw/sig_wood/tar.jpg'
    painter = Painter()
    masker = Masker(source)
    masked_source = masker.edit(painter, 'source')
    masker_mover = MaskMover(masked_source, target)
    offset, scale_ratio = masker_mover.edit(painter, 'target')
    blender = PoissonBlender(masked_source, target, offset, scale_ratio)
    blender.blend(painter, 'default')
    blender.set_div_of_guidance_field('mix')
    blender.blend(painter, 'mix')
    blender.set_div_of_guidance_field('average')
    blender.blend(painter, 'average')
    painter.erase_all()

if __name__ == '__main__':
    main(sys.argv)