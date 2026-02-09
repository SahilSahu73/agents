from slowapi import Limiter
from slowapi.util import get_remote_address
from app.core.config import settings

# Rate Limiter Configuration
# we need to define how we identify a unique user -> IP address
# and then apply the default limits defined in our settings earlier (env)

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=list(settings.RATE_LIMIT_DEFAULT)
)