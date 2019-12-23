import numpy as np
from PIL import Image
from torchvision.transforms import Resize
import os
import moviepy.editor as mpy


def read_gif(im, resize=None, transpose=True):
    if isinstance(im, str):
        im = Image.open(im)
        
    im.seek(0)

    frames = []
    try:
        while 1:
            frame = im.convert('RGB')
            if resize is not None:
                frame = frame.resize((32,32))
                
            frames.append(np.array(frame))
            im.seek(im.tell()+1)
    except EOFError:
        pass # end of sequence
    
    image = np.stack(frames, axis=0)
    if transpose:
        image = np.transpose(image, [0, 3, 1, 2])
    return image


def npy_to_gif(im_list, filename, fps=5):
    save_dir = '/'.join(str.split(filename, '/')[:-1])

    if not os.path.exists(save_dir):
        print('creating directory: ', save_dir)
        os.makedirs(save_dir)

    if isinstance(im_list, np.ndarray):
        im_list = list(im_list)

    clip = mpy.ImageSequenceClip(im_list, fps=fps)
    clip.write_gif(filename + '.gif')


def npy_to_mp4(im_list, filename, fps=4):
    save_dir = '/'.join(str.split(filename, '/')[:-1])

    if save_dir and not os.path.exists(save_dir):
        print('creating directory: ', save_dir)
        os.mkdir(save_dir)

    clip = mpy.ImageSequenceClip(im_list, fps=fps)
    clip.write_videofile(filename + '.mp4')
    

npy2gif = npy_to_gif
npy2mp4 = npy_to_mp4


def ch_first2last(video):
    return video.transpose((0,2,3,1))


def ch_last2first(video):
    return video.transpose((0,3,1,2))


def resize_video_chfirst(video, size):
    return ch_last2first(resize_video(ch_first2last(video), size))


def resize_video(video, size):
    # return video
    # video = ch_first2last(video).astype(np.uint8)
    # lazy transform ;)
    # transformed_video = video
    
    # moviepy is memory inefficient
    # clip = mpy.ImageSequenceClip(np.split(images, images.shape[0]), fps=1)
    # clip = clip.resize((self.img_sz, self.img_sz))
    # images = np.array([frame for frame in clip.iter_frames()])
    
    # looping over time is too slow for long videos
    transformed_video = np.stack([np.asarray(Resize(size)(Image.fromarray(im))) for im in video], axis=0)
    
    # Using pytorch is also slow
    # x = torch.from_numpy(video.transpose((0,3,1,2))).float()
    # x = F.interpolate(x, size)
    # transformed_video = x.data.numpy().transpose((0,2,3,1))
    
    # It seems that cv2 can resize images with up to 512 channels, which is what this implementation uses
    # It is as fast as the looping implementation above
    # sh = list(video.shape)
    # x = video.transpose((1, 2, 3, 0)).reshape(sh[1:3]+[-1])
    # n_split = math.ceil(x.shape[2] / 512.0)
    # x = np.array_split(x, n_split, 2)
    # x = np.concatenate([cv2.resize(im, size) for im in x], 2)
    # transformed_video = x.reshape(list(size) + [sh[3], sh[0]]).transpose((3, 0, 1, 2))
    
    # scipy.ndimage.zoom seems to be even slower than looping. Wow.
    # sh = [video.shape[1] / size[0], video.shape[2] / size[1]]
    # transformed_video = ndimage.zoom(video, [1] + sh + [1])
    
    # transformed_video = ch_last2first(transformed_video).astype(np.float32)
    return transformed_video