import streamlit as st
from functools import wraps
from typing import Any, Callable, Optional


def st_cache_data_with_ttl(ttl: int = 3600):
    """
    Streamlit cache decorator with TTL (time-to-live) in seconds.
    Default TTL is 1 hour (3600 seconds).
    """
    def decorator(func: Callable) -> Callable:
        @st.cache_data(ttl=ttl)
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            return func(*args, **kwargs)
        return wrapper
    return decorator


def st_cache_resource_with_ttl(ttl: int = 3600):
    """
    Streamlit cache resource decorator with TTL.
    Use for database connections and other resources.
    """
    def decorator(func: Callable) -> Callable:
        @st.cache_resource(ttl=ttl)
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            return func(*args, **kwargs)
        return wrapper
    return decorator


def clear_all_cache():
    """Clear all Streamlit caches."""
    st.cache_data.clear()
    st.cache_resource.clear()


def cache_key_from_date(date_obj) -> str:
    """Generate a consistent cache key from a date object."""
    return date_obj.strftime("%Y-%m-%d")


# Commonly used cache decorators
cache_short = st_cache_data_with_ttl(300)   # 5 minutes
cache_medium = st_cache_data_with_ttl(1800)  # 30 minutes
cache_long = st_cache_data_with_ttl(3600)    # 1 hour