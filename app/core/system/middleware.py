"""Custom middleware for tracking metrics and other cross-cutting concerns."""

import time
from typing import Callable

from fastapi import Request
from jose import (
    JWTError,
    jwt,
)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from app.core.system.config import settings
from app.core.system.logging import (
    bind_context,
    clear_context,
)
from app.core.system.telemetry import (
    http_request_duration_seconds,
    http_requests_total,
)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for tracking HTTP request metrics, i.e. request duration and status codes."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Track metrics for each request.

        Args:
            request: The incoming request
            call_next: The next middleware or route handler

        Returns:
            Response: The response from the application
        """
        start_time = time.time()
        status_code = 500

        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception:
            # If the app crashes, we still want to record the 500 error.
            raise
        finally:
            duration = time.time() - start_time

            # Record metrics to prometheus
            # We filter out /metrics and /health to avoid noise
            if request.url.path not in ["/metrics", "/health"]:
                http_requests_total.labels(
                    method=request.method, 
                    endpoint=request.url.path, 
                    status=status_code
                ).inc()

                http_request_duration_seconds.labels(
                    method=request.method, 
                    endpoint=request.url.path
                ).observe(duration)

        return response


class LoggingContextMiddleware(BaseHTTPMiddleware):
    """
    Middleware for adding user_id and session_id to logging context.
    Extracts User IDs from JWTs before the request hits the router.
    This ensures that even authentication errors are logged with the correct context.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Extract user_id and session_id from authenticated requests and add to logging context.

        Args:
            request: The incoming request
            call_next: The next middleware or route handler

        Returns:
            Response: The response from the application
        """
        try:
            # Clear any existing context from previous requests
            clear_context()

            # Extract token from Authorization header
            auth_header = request.headers.get("authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]

                try:
                    # Decode token to get session_id (stored in "sub" claim)
                    payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
                    session_id = payload.get("sub")

                    if session_id:
                        # Bind session_id to logging context
                        bind_context(session_id=session_id)

                        # Try to get user_id from request state after authentication
                        # This will be set by the dependency injection if the endpoint uses authentication
                        # We'll check after the request is processed

                except JWTError:
                    # Token is invalid, but don't fail the request - let the auth dependency handle it
                    pass

            # Process the request
            response = await call_next(request)

            # After request processing, check if user info was added to request state
            if hasattr(request.state, "user_id"):
                bind_context(user_id=request.state.user_id)

            return response

        finally:
            # Always clear context after request is complete to avoid leaking to other requests
            clear_context()
