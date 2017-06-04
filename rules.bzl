def pyx_library(name, srcs):
  outs = [x.split(".")[0] + ".so" for x in srcs]
  cmd_head = ("python -c 'import sys;" + 
      "sys.argv = [\"-c\", \"build_ext\", \"--inplace\"];" + 
      "from distutils.core import setup;" + 
      "from distutils.extension import Extension;" +
      "from Cython.Distutils import build_ext;" + 
      "import numpy;" +
      "import glob;" + 
      "import shutil;")
  cmd_setup = []
  for ss in srcs:
    basename = ss.split(".")[0]
    cmd_setup.append(
      #"shutil.rmtree(\"" + basename + ".so\");" + 
      "setup(" +
      "cmdclass={\"build_ext\": build_ext}," +
      "ext_modules=[" +
      "Extension(" +
      "\"" + basename +"\", sources=[\"'$(SRCS)'\"], " +
      "include_dirs=[numpy.get_include()])" +
      "],);")
    cmd_setup.append(
      "old_file = glob.glob(\"" + basename + "*.so\")[0];" +
      "print(old_file);" +
      #"shutil.move(old_file, \"" + basename + ".so\");"
      "shutil.move(old_file, \"'$@'\");"
    )
  cmd = cmd_head + "".join(cmd_setup) + """'"""
  #print(cmd)
  native.genrule(
    name = name + "_build",
    srcs = srcs,
    outs = outs,
    cmd = cmd,
  )
  native.py_library(name=name, srcs=[], data=outs)
