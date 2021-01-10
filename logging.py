import numpy as np
from PIL import Image


def fig2img(fig):
  """Converts a given figure handle to a 3-channel numpy image array."""
  fig.canvas.draw()
  w, h = fig.canvas.get_width_height()
  buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
  buf.shape = (w, h, 4)
  buf = np.roll(buf, 3, axis=2)
  w, h, d = buf.shape
  return np.array(Image.frombytes("RGBA", (w, h), buf.tostring()), dtype=np.float32)[:, :, :3] / 255.
