# \__future\__模块

python提供了`__fututure`模块，把下一个新版本的特性导入当前版本

在python3.x中，所有的除法都是精确除法，地板除用`//`表示：
如果是整数相除，结果仍是整数，余数会被扔掉，这种除法叫“地板除”
```python
10 / 3 = 3.33333333333
10.0 / 3 = 3.33333333333
10 // 3 = 3
```

- 在开头加上`from __future__ import print_function`这句之后，即使在python2.X，使用print就得像python3.X那样加括号使用。python2.X中print不需要括号，而在python3.X中则需要。
- 在python2 中导入未来的支持的语言特征中division(精确除法)，即`from __future__ import division `，当我们在程序中没有导入该特征时，"/"操作符执行的只能是整除，也就是取整数，只有当我们导入division(精确算法)以后，"/"执行的才是精确算法。
- 假设当前你的文件夹的状态是这样的：
project
	main.py
	numpy.py

当在main.py中的开头使用：
```python
import numpy as np
```
则会优先寻找并导入当前目录下的numpy文件。

而如果你真正想的是导入标准的numpy库，则需要写成：
```python
from __future__ import absolute_import
import numpy as np
```
如果你已经加上了absolute_import，但是想导入当前目录下的numpy文件，则需要写成：
```python
from __future__ import absolute_import
from project import numpy as np
```

