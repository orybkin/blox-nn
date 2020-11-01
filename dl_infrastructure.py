import sys
import os
import pipes
import pathlib


def save_cmd(base_dir):
  if not isinstance(base_dir, pathlib.Path):
    base_dir = pathlib.Path(base_dir)
  train_cmd = 'python ' + ' '.join([sys.argv[0]] + [pipes.quote(s) for s in sys.argv[1:]])
  train_cmd += '\n'
  print('\n' + '*' * 80)
  print('Training command:\n' + train_cmd)
  print('*' * 80 + '\n')
  with open(base_dir / "cmd.txt", "w") as f:
    f.write(train_cmd)


def save_git(base_dir):
  # save code revision
  print('Save git commit and diff to {}/git.txt'.format(base_dir))
  cmds = ["echo `git rev-parse HEAD` > {}".format(
    os.path.join(base_dir, 'git.txt')),
    "git diff >> {}".format(
      os.path.join(base_dir, 'git.txt'))]
  print(cmds)
  os.system("\n".join(cmds))
