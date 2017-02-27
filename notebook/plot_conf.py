# matplotlib and stuff
import matplotlib.pyplot as plt
from matplotlib import rc
#%matplotlib inline  # not from script
get_ipython().run_line_magic('matplotlib', 'inline')

# configurations for dark background
plt.style.use(['dark_background', 'bmh'])

# configuration for bright background
# plt.style.use('bmh')

# remove background colour, set figure size
rc('figure', figsize=(16, 8), facecolor='none', max_open_warning=False)
rc('axes', facecolor='none')