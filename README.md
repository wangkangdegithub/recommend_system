## 菜品推荐



目录结构
```
recommend_system:.
│  README.md
│
│
├─algorithm   
│  │  pkg.py  (工具函数)
│  │  svd.py  (SVD算法)
│  │  __init__.py
│
├─data
│      用户给菜品打分表.xlsx
│
└─run
        metric.py  (推荐算法评估)
        recommend.py  (推荐算法推荐)
        __init__.py
```

**开发环境:**
- windows-10 
- Python 3.6.1 
- pandas==0.24.0 
- numpy==1.17.3



**代码执行：**

>python recommend_system\run\recommend.py
