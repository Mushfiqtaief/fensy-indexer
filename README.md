# Fensy Indexer & ML Vector Compiler

Automated cloud indexer compiling BDIX and FTP streaming sources into portable, pre-indexed SQLite databases with 64-dimensional semantic recommendation vectors for the **Fensy** streaming client.

## Features
- **Automated 6-Hour Scheduled Crawling**: Runs via GitHub Actions cron pipeline.
- **Embedded BDIX Proxy Routing**: Uses embedded Mihomo daemon configured with BDIX SOCKS5 proxies to bypass regional ISP network boundaries.
- **On-Device Recommendation Vectors**: Computes 64-dimensional dense semantic vectors per title using TF-IDF & SVD projection, stored directly as 256-byte BLOBs in SQLite.
- **Instant Client Performance**: Clients mount the compiled database locally in `< 5ms`, delivering instant offline catalog browsing and `< 2ms` SQLite FTS5 search.

## Published CDN Release Artifacts
Published to the [`latest` release tag](https://github.com/Mushfiqtaief/fensy-indexer/releases/tag/latest):
- `circleftp.db.zst`: Compressed SQLite database (~5 MB).
- `circleftp.db.sha256`: 64-byte digest for fast conditional updates.
- `manifest.json`: Index version, item count, and CDN URLs.
