# _*_ coding:utf-8 _*_

from collections import namedtuple
#创建了一个tuple对象
Point=namedtuple('Point',['x','y'])
p=Point(1,2)
print(p.x)
print(p.y)


from collections import deque

#实现了队列的对象
q=deque(['a','b','c'])
q.append('x')
q.appendleft('y')
print(q)
q.pop()
q.popleft()
print(q)


from collections import defaultdict

#为dict的key设置默认值
dd=defaultdict(int)
dd['key1']=dd['key1']+1
print(dd.keys())


from collections import OrderedDict
#OrderDict会按照插入顺序排序,不过并没有看出来和普通的有什么区别
d = dict([('n', 1), ('b', 2), ('c', 3)])
print(d)

d=OrderedDict([('n',1), ('b', 2), ('c', 3)])
print(d)


from collections import Counter
#Counter计数器
c=Counter()
for ch in 'programming':
    c[ch]=c[ch]+1

print(c)

