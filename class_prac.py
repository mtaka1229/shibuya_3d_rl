from typing import Any


class Person: #人間を作る設計書のようなもの
    # class内で使用される関数を特にメソッドと言う
    def __init__(self, name, nationality, age):  #classに含まれてる特殊な関数その1（初期化する関数）
        self.name = name
        self.nationality = nationality
        self.age = age
    
    def __call__(self): #init関数と並ぶ，classの特殊な関数（メソッド）
        print('ここはcall関数です')
      
    # 引数にselfを入れるのを忘れないように
    def say_hello(self, name):
        print('こんにちは．{}さん．私は{}です'.format(name, self.name)) # format funcとはなんだ！！！

# インスタンス化（実体化）
matsunaga = Person('Matsunaga', 'JPN', 22)

#print(matsunaga.name, matsunaga.nationality, matsunaga.age, matsunaga.say_hello('陸上部'))

#mike = Person('Mike', 'US', 13)

# この呼び出し方は正しくない
#matsunaga.__call__()

# instance()で呼び出せるのがcall関数！！
matsunaga()

