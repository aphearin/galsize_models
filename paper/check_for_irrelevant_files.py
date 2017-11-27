import os
import fnmatch
import subprocess


def fname_generator(root_dirname, basename_filepat):
    """ Yield the absolute path of all files in the directory tree of ``root_dirname``
    with a basename matching the input pattern
    """

    for path, dirlist, filelist in os.walk(root_dirname):
        for filename in fnmatch.filter(filelist, basename_filepat):
            yield filename


figure_names = list(fname_generator(os.path.dirname(os.path.realpath(__file__)), '*.pdf'))

absent_fignames = []
used_fignames = []
command = 'grep "{0}" galsize_paper.tex'
for figname in figure_names:
    result = os.system(command.format(figname))
    if result is 0:
        used_fignames.append(figname)
    else:
        absent_fignames.append(figname)

remove_command = 'git rm FIGS/{0}'
for figname in absent_fignames:
    result = os.system(remove_command.format(figname))
