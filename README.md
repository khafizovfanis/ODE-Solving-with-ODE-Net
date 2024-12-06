Для работы требуется установка библиотеки [torchdiffeq](https://github.com/rtqichen/torchdiffeq):
```
pip install torchdiffeq
```

Запуск эксперимента производится командой

```
python code/ode.py --viz --point_type=saddle --noise_scale=0.1
```

В аргументе `point_type` указывается тип особой точки, поддерживаемые: `['uniform', 'saddle', 'center', 'spiral']`,`noise_scale` задает дисперсию шума.
