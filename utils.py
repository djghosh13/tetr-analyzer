Identity = lambda obj: obj

class _PropertyGetter(object):
    def __init__(self, fn=Identity):
        self._fn = fn

    def __getattr__(self, key):
        return _PropertyGetter(lambda obj: getattr(self._fn(obj), key))

    def __getitem__(self, key):
        return _PropertyGetter(lambda obj: self._fn(obj)[key])

    def __add__(self, other):
        if isinstance(other, _PropertyGetter):
            return _PropertyGetter(lambda obj: self._fn(obj) + other._fn(obj))
        return _PropertyGetter(lambda obj: self._fn(obj) + other)

    def __sub__(self, other):
        if isinstance(other, _PropertyGetter):
            return _PropertyGetter(lambda obj: self._fn(obj) - other._fn(obj))
        return _PropertyGetter(lambda obj: self._fn(obj) - other)

    def __eq__(self, other):
        if isinstance(other, _PropertyGetter):
            return _PropertyGetter(lambda obj: self._fn(obj) == other._fn(obj))
        return _PropertyGetter(lambda obj: self._fn(obj) == other)

    def __ne__(self, other):
        if isinstance(other, _PropertyGetter):
            return _PropertyGetter(lambda obj: self._fn(obj) != other._fn(obj))
        return _PropertyGetter(lambda obj: self._fn(obj) != other)

    def __call__(self, obj):
        return self._fn(obj)

Get = _PropertyGetter()