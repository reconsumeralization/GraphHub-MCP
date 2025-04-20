import inspect
from collections.abc import Callable
from typing import Generic, ParamSpec, TypeVar, cast, overload

from typing_extensions import Self

R = TypeVar("R")
P = ParamSpec("P")


class DecoratedFunction(Generic[P, R]):
    """
    Descriptor for decorated functions.

    This class enables decorators to work seamlessly across:
    - Plain functions
    - Instance methods
    - Class methods (decorator must be applied before @classmethod)
    - Static methods
    - Both sync and async functions

    It stores the original function and can be used to attach metadata for tool registration.

    Example usage:

        def my_decorator(fn: Callable[P, R]) -> DecoratedFunction[P, R]:
            return DecoratedFunction(fn)

    On a function:
        @my_decorator
        def my_function(a: int, b: int) -> int:
            return a + b

    On an instance method:
        class Test:
            @my_decorator
            def my_function(self, a: int, b: int) -> int:
                return a + b

    On a class method:
        class Test:
            @classmethod
            @my_decorator
            def my_function(cls, a: int, b: int) -> int:
                return a + b

        # NOTE: Decorator must be applied before @classmethod!

    On a static method:
        class Test:
            @staticmethod
            @my_decorator
            def my_function(a: int, b: int) -> int:
                return a + b
    """

    fn: Callable[P, R]

    def __init__(self, fn: Callable[P, R]) -> None:
        self.fn = fn

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """
        Call the original function.

        Raises:
            TypeError: If used incorrectly with classmethod decorator order.
        """
        try:
            return self.fn(*args, **kwargs)
        except TypeError as e:
            # Provide a clear error if decorator order is wrong for classmethods
            if "'classmethod' object is not callable" in str(e):
                raise TypeError(
                    "To apply this decorator to a classmethod, apply the decorator first, then @classmethod on top."
                ) from e
            raise

    @overload
    def __get__(self, instance: None, owner: type | None = None) -> Self: ...
    @overload
    def __get__(self, instance: object, owner: type | None = None) -> Callable[P, R]: ...

    def __get__(
        self, instance: object | None, owner: type | None = None
    ) -> Self | Callable[P, R]:
        """
        Descriptor protocol: return self when accessed from the class,
        or the function bound to the instance when accessed from an instance.
        """
        if instance is None:
            return self
        # Return the original function bound to the instance
        # TODO: Consider supporting binding for classmethod/staticmethod if needed
        return cast(Callable[P, R], self.fn.__get__(instance, owner))

    def __repr__(self) -> str:
        """
        Return a representation that matches Python's function representation.
        """
        module = getattr(self.fn, "__module__", "unknown")
        qualname = getattr(self.fn, "__qualname__", str(self.fn))
        sig_str = str(inspect.signature(self.fn))
        return f"<function {module}.{qualname}{sig_str}>"

# TODO: Add unit tests for edge cases (classmethod/staticmethod, async functions, etc.)
# TODO: Consider supporting functools.wraps for better introspection
