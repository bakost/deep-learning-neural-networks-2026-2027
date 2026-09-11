"""Мини-фреймворк контрактного программирования (Design by Contract).

В Python нет встроенной поддержки контрактов в стиле Eiffel/Ada, поэтому
здесь реализован минимальный, но полноценный аналог популярных библиотек
``icontract`` и ``deal``: декораторы :func:`require` (предусловие),
:func:`ensure` (постусловие) и :func:`invariant` (инвариант класса).

Ключевые свойства
-----------------
* **Предикат — обычная лямбда** от *имён параметров* декорируемой функции.
  Постусловие дополнительно может принимать ``result`` (возвращённое
  значение) и ``OLD`` (значения аргументов «до вызова»).
* **Контракты отключаемы.** Они выключены, если интерпретатор запущен с
  ``-O`` (``__debug__ == False``) или если переменная окружения
  ``JACOBI_CONTRACTS`` равна ``0``/``off``/``false``. Управлять ими можно и
  во время выполнения: :func:`enable`, :func:`disable`, :func:`contracts_disabled`.
* **Нулевая цена в горячем цикле.** Контракты навешиваются только на
  публичный API и вспомогательные функции, но не на внутренний цикл вращений.

Пример
------
>>> from jacobi_eigen.contracts import require, ensure
>>> @require(lambda n: n >= 0, "n должно быть неотрицательным")
... @ensure(lambda result: result >= 0, "факториал неотрицателен")
... def factorial(n):
...     return 1 if n == 0 else n * factorial(n - 1)
>>> factorial(5)
120
>>> factorial(-1)
Traceback (most recent call last):
    ...
jacobi_eigen.exceptions.PreconditionViolation: ...
"""

from __future__ import annotations

import functools
import inspect
import os
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, Optional, TypeVar, cast

from .exceptions import (
    InvariantViolation,
    PostconditionViolation,
    PreconditionViolation,
)

__all__ = [
    "require",
    "ensure",
    "invariant",
    "enabled",
    "enable",
    "disable",
    "contracts_disabled",
]

F = TypeVar("F", bound=Callable[..., Any])

_OFF_VALUES = {"0", "off", "false", "no", ""}

#: Глобальный флаг: включены ли проверки контрактов.
_ENABLED: bool = __debug__ and os.environ.get("JACOBI_CONTRACTS", "1").lower() not in _OFF_VALUES


def enabled() -> bool:
    """Вернуть ``True``, если проверки контрактов сейчас активны."""
    return _ENABLED


def enable() -> None:
    """Включить проверки контрактов."""
    global _ENABLED
    _ENABLED = True


def disable() -> None:
    """Выключить проверки контрактов (например, для замеров производительности)."""
    global _ENABLED
    _ENABLED = False


@contextmanager
def contracts_disabled() -> Iterator[None]:
    """Контекстный менеджер: временно отключить контракты.

    >>> with contracts_disabled():
    ...     pass  # здесь предусловия не проверяются
    """
    global _ENABLED
    previous = _ENABLED
    _ENABLED = False
    try:
        yield
    finally:
        _ENABLED = previous


def _describe(predicate: Callable[..., Any], description: Optional[str]) -> str:
    """Получить человекочитаемое описание предиката."""
    if description:
        return description
    try:
        source = inspect.getsource(predicate).strip()
    except (OSError, TypeError):  # pragma: no cover - интерактивная сессия
        source = getattr(predicate, "__name__", repr(predicate))
    return " ".join(source.split())


def _collect(
    parameter_names: "tuple[str, ...]",
    bound: inspect.BoundArguments,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Собрать аргументы для предиката по именам его параметров.

    Имена вычисляются один раз при декорировании и передаются сюда готовыми:
    вызов ``inspect.signature`` на каждой проверке обходился бы в разы дороже
    самой проверки, а предусловия вызываются в том числе из внутреннего
    цикла метода.
    """
    available = dict(bound.arguments)
    if extra:
        available.update(extra)
    kwargs: Dict[str, Any] = {}
    for name in parameter_names:
        if name not in available:
            raise TypeError(
                f"Контракт ссылается на неизвестный параметр {name!r}; "
                f"доступны: {sorted(available)}"
            )
        kwargs[name] = available[name]
    return kwargs


def require(predicate: Callable[..., Any], description: Optional[str] = None) -> Callable[[F], F]:
    """Объявить **предусловие** функции.

    Parameters
    ----------
    predicate:
        Функция (обычно лямбда), параметры которой называются так же, как
        параметры декорируемой функции. Должна вернуть истинное значение.
    description:
        Пояснение, попадающее в текст ошибки.

    Raises
    ------
    PreconditionViolation
        Если предикат вернул ложное значение.
    """

    def decorator(func: F) -> F:
        signature = inspect.signature(func)
        parameter_names = tuple(inspect.signature(predicate).parameters)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if _ENABLED:
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                if not predicate(**_collect(parameter_names, bound)):
                    raise PreconditionViolation(
                        f"Нарушено предусловие функции {func.__qualname__}: "
                        f"{_describe(predicate, description)}"
                    )
            return func(*args, **kwargs)

        wrapper.__contract_require__ = getattr(func, "__contract_require__", []) + [  # type: ignore[attr-defined]
            _describe(predicate, description)
        ]
        return cast(F, wrapper)

    return decorator


def ensure(predicate: Callable[..., Any], description: Optional[str] = None) -> Callable[[F], F]:
    """Объявить **постусловие** функции.

    Предикат может принимать любые параметры функции, а также специальное имя
    ``result`` — значение, которое функция собирается вернуть.

    Raises
    ------
    PostconditionViolation
        Если предикат вернул ложное значение.
    """

    def decorator(func: F) -> F:
        signature = inspect.signature(func)
        parameter_names = tuple(inspect.signature(predicate).parameters)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not _ENABLED:
                return func(*args, **kwargs)
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            result = func(*args, **kwargs)
            if not predicate(**_collect(parameter_names, bound, {"result": result})):
                raise PostconditionViolation(
                    f"Нарушено постусловие функции {func.__qualname__}: "
                    f"{_describe(predicate, description)}"
                )
            return result

        wrapper.__contract_ensure__ = getattr(func, "__contract_ensure__", []) + [  # type: ignore[attr-defined]
            _describe(predicate, description)
        ]
        return cast(F, wrapper)

    return decorator


def invariant(
    predicate: Callable[..., Any], description: Optional[str] = None
) -> Callable[[type], type]:
    """Объявить **инвариант класса**.

    Предикат принимает единственный параметр ``self`` и проверяется после
    ``__init__`` и после каждого публичного метода (не начинающегося с ``_``).

    Raises
    ------
    InvariantViolation
        Если предикат вернул ложное значение.
    """

    def decorator(cls: type) -> type:
        text = _describe(predicate, description)

        def check(instance: Any) -> None:
            if _ENABLED and not predicate(instance):
                raise InvariantViolation(
                    f"Нарушен инвариант класса {cls.__name__}: {text}"
                )

        def wrap(method: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(method)
            def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
                result = method(self, *args, **kwargs)
                check(self)
                return result

            return wrapper

        for name, member in list(vars(cls).items()):
            if name.startswith("_") and name != "__init__":
                continue
            if isinstance(member, (staticmethod, classmethod, property)):
                continue
            if inspect.isfunction(member):
                setattr(cls, name, wrap(member))

        cls.__contract_invariants__ = getattr(cls, "__contract_invariants__", []) + [text]  # type: ignore[attr-defined]
        return cls

    return decorator
