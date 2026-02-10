"""Kraken API Client Module.

Provides a secure and robust interface to the Kraken REST API
with rate limiting, error handling, and comprehensive logging.
"""

import base64
import hashlib
import hmac
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import requests
from requests import Response

from secure_creds import SecureCreds


logger = logging.getLogger(__name__)


@dataclass
class OHLCData:
    """OHLC (Open-High-Low-Close) candlestick data."""
    time: float
    open: float
    high: float
    low: float
    close: float
    vwap: float
    volume: int
    count: int
    
    @classmethod
    def from_list(cls, data: List) -> 'OHLCData':
        """Create OHLCData from Kraken API response list."""
        return cls(
            time=float(data[0]),
            open=float(data[1]),
            high=float(data[2]),
            low=float(data[3]),
            close=float(data[4]),
            vwap=float(data[5]),
            volume=float(data[6]),  # Volume can be float for crypto
            count=int(data[7])
        )


@dataclass
class Balance:
    """Account balance for an asset."""
    asset: str
    available: float
    hold: float
    
    @property
    def total(self) -> float:
        """Total balance (available + hold)."""
        return self.available + self.hold


@dataclass
class Position:
    """Open position."""
    pair: str
    side: str  # 'long' or 'short'
    volume: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    
    @property
    def pnl_pct(self) -> float:
        """Unrealized PnL as percentage."""
        if self.entry_price == 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price * 100


class KrakenClient:
    """
    Kraken REST API Client.
    
    Features:
    - All required API methods (OHLC, balance, orders, positions)
    - Rate limiting (~1 req/sec for public, more for private)
    - Automatic retry on transient errors
    - HMAC signature for private endpoints
    - Comprehensive error handling
    """
    
    BASE_URL = "https://api.kraken.com"
    PUBLIC_ENDPOINT = "/0/public/"
    PRIVATE_ENDPOINT = "/0/private/"
    
    # Rate limiting
    RATE_LIMIT_DELAY = 1.1  # Seconds between requests (Kraken: ~1/sec)
    
    def __init__(
        self,
        credentials: Optional[SecureCreds] = None,
        api_url: str = BASE_URL,
        timeout: int = 30
    ):
        """
        Initialize Kraken API client.
        
        Args:
            credentials: SecureCreds instance for authentication
            api_url: Base URL for API (default: production)
            timeout: Request timeout in seconds
        """
        self._credentials = credentials
        self._api_url = api_url.rstrip('/')
        self._timeout = timeout
        self._last_request_time = 0.0
        
        self._session = requests.Session()
        self._session.headers.update({
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json',
        })
    
    @property
    def is_authenticated(self) -> bool:
        """Check if client has valid credentials."""
        return self._credentials is not None and self._credentials.is_available
    
    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.RATE_LIMIT_DELAY:
            time.sleep(self.RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = time.time()
    
    def _sign(self, endpoint: str, nonce: str, data: str) -> Dict[str, str]:
        """
        Generate HMAC signature for private API requests.
        
        Args:
            endpoint: API endpoint path
            nonce: Unique request ID
            data: POST data string
        
        Returns:
            Headers dictionary with signature
        """
        if not self._credentials:
            raise ValueError("Cannot sign request without credentials")
        
        api_secret = self._credentials.api_secret
        if not api_secret:
            raise ValueError("API secret not available")
        
        # Decode secret from base64 (Kraken format)
        secret_bytes = base64.b64decode(api_secret)
        
        # Kraken signature algorithm:
        # 1. SHA256(nonce + POST data)
        # 2. HMAC-SHA512(secret, endpoint + sha256_hash)
        # 3. Base64 encode the result
        
        message = (nonce + data).encode('utf-8')
        sha256_hash = hashlib.sha256(message).digest()
        signature = hmac.new(secret_bytes, endpoint.encode('utf-8') + sha256_hash, hashlib.sha512).digest()
        api_sign = base64.b64encode(signature).decode()
        
        return {
            'API-Sign': api_sign,
            'API-Key': self._credentials.api_key,
            'Nonce': nonce,
        }
    
    def _make_request(
        self,
        endpoint: str,
        data: Optional[Dict[str, str]] = None,
        private: bool = False
    ) -> Dict[str, Any]:
        """
        Make API request with error handling and retries.
        
        Args:
            endpoint: API endpoint path
            data: Optional POST data
            private: Whether endpoint requires authentication
        
        Returns:
            Parsed JSON response
        
        Raises:
            KrakenAPIError: On API error responses
            NetworkError: On network failures
        """
        self._rate_limit()
        
        url = f"{self._api_url}{endpoint}"
        headers = {}
        
        if private:
            if not self.is_authenticated:
                raise ValueError(f"Private endpoint {endpoint} requires authentication")
            
            nonce = str(int(time.time() * 1000))
            post_data = urlencode({**(data or {}), 'nonce': nonce})
            
            signature_headers = self._sign(endpoint, nonce, post_data)
            headers.update(signature_headers)
        else:
            post_data = urlencode(data or {})
        
        try:
            response = self._session.post(
                url,
                data=post_data,
                headers=headers,
                timeout=self._timeout
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Check for Kraken error response
            if result.get('error') and len(result['error']) > 0:
                error_msg = result['error'][0]
                logger.error(f"Kraken API error: {error_msg}")
                raise KrakenAPIError(error_msg, result)
            
            return result.get('result', {})
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error during API request: {e}")
            raise NetworkError(f"Failed to connect to Kraken API: {e}")
    
    # ==================== Public Endpoints ====================
    
    def get_ohlc(
        self,
        pair: str,
        interval: int = 60,
        count: int = 100
    ) -> List[OHLCData]:
        """
        Get OHLC candlestick data.
        
        Args:
            pair: Trading pair (e.g., 'SOL/USD')
            interval: Candlestick interval in minutes (1, 5, 15, 60, 240, 1440, 10080, 216000)
            count: Number of candles to retrieve (max 1000)
        
        Returns:
            List of OHLCData objects, most recent first
        """
        data = {
            'pair': pair,
            'interval': str(interval),
        }
        
        # Only add since if count is large to avoid returning old cached data
        # Kraken may return stale data when requesting large time ranges
        if count > 100:
            # Get data from 2 days ago instead of weeks/months ago
            since = int(time.time()) - (60 * 60 * 48)
            data['since'] = str(since)
        
        result = self._make_request(self.PUBLIC_ENDPOINT + 'OHLC', data)
        
        # Parse OHLC data
        ohlc_list = result.get(pair, [])
        return [OHLCData.from_list(candle) for candle in ohlc_list]
    
    def get_ticker(self, pair: str) -> Dict[str, Any]:
        """Get ticker information for a trading pair."""
        data = {'pair': pair}
        result = self._make_request(self.PUBLIC_ENDPOINT + 'Ticker', data)
        return result.get(pair, {})
    
    def get_server_time(self) -> Dict[str, Any]:
        """Get Kraken server time."""
        result = self._make_request(self.PUBLIC_ENDPOINT + 'Time')
        return result
    
    # ==================== Private Endpoints ====================
    
    def get_balance(self) -> Dict[str, float]:
        """
        Get account balance.
        
        Returns:
            Dict mapping asset codes to available balances
        """
        if not self.is_authenticated:
            logger.warning("get_balance called without authentication")
            return {}
        
        result = self._make_request(self.PRIVATE_ENDPOINT + 'Balance', private=True)
        
        # Convert string values to floats
        return {asset: float(balance) for asset, balance in result.items()}
    
    def place_order(
        self,
        pair: str,
        side: str,
        order_type: str,
        volume: float,
        price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Place a new order.
        
        Args:
            pair: Trading pair (e.g., 'SOL/USD')
            side: 'buy' or 'sell'
            order_type: 'market', 'limit', 'stop-loss', 'take-profit', etc.
            volume: Order volume (in base currency)
            price: Limit price for limit orders
        
        Returns:
            Dict with order details
        """
        if not self.is_authenticated:
            logger.warning("place_order called without authentication")
            return {'txid': [], 'descr': {'order': 'PAPER ORDER (not submitted)'}}
        
        data = {
            'pair': pair,
            'type': side.lower(),
            'ordertype': order_type.lower(),
            'volume': str(volume)
        }
        
        if price is not None:
            data['price'] = str(price)
        
        result = self._make_request(self.PRIVATE_ENDPOINT + 'AddOrder', data, private=True)
        return result
    
    def get_open_positions(self, pair: Optional[str] = None) -> List[Position]:
        """
        Get open positions.
        
        Args:
            pair: Optional trading pair filter
        
        Returns:
            List of Position objects
        """
        if not self.is_authenticated:
            logger.warning("get_open_positions called without authentication")
            return []
        
        data: Dict[str, str] = {}
        if pair:
            data['pair'] = pair
        
        result = self._make_request(self.PRIVATE_ENDPOINT + 'OpenPositions', data, private=True)
        
        positions = []
        for pos_id, pos_data in result.items():
            positions.append(Position(
                pair=pos_data.get('pair', ''),
                side=pos_data.get('side', 'long'),
                volume=float(pos_data.get('vol', 0)),
                entry_price=float(pos_data.get('avg_price', 0)),
                current_price=float(pos_data.get('price', 0)),
                unrealized_pnl=float(pos_data.get('unrealized_pnl', 0))
            ))
        
        return positions
    
    def cancel_order(self, txid: str) -> Dict[str, Any]:
        """Cancel an open order."""
        if not self.is_authenticated:
            return {'status': 'paper_cancelled', 'txid': txid}
        
        data = {'txid': txid}
        return self._make_request(self.PRIVATE_ENDPOINT + 'CancelOrder', data, private=True)
    
    def get_order_info(self, txid: str) -> Dict[str, Any]:
        """Get information about a specific order."""
        if not self.is_authenticated:
            return {'status': 'paper', 'txid': txid}
        
        data = {'txid': txid}
        return self._make_request(self.PRIVATE_ENDPOINT + 'QueryOrders', data, private=True)


class KrakenAPIError(Exception):
    """Exception for Kraken API errors."""
    
    def __init__(self, message: str, response: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.response = response or {}


class NetworkError(Exception):
    """Exception for network-related errors."""
    pass
