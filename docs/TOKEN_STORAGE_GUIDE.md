# Token Storage Best Practices

## ✅ Recommended Approach: User-Level Storage

**Location:** `/home/username/.qtrader/schwab_tokens.json`

### Why This Is Best

1. **User-Level Scope**

   - Tokens are personal credentials (tied to your Schwab account)
   - Not project-specific - same credentials work across all projects
   - Natural fit for user home directory storage

1. **Follows Industry Conventions**

   - Similar to: `~/.ssh/`, `~/.aws/`, `~/.docker/`, `~/.kube/`
   - Standard location for personal authentication credentials
   - Expected by developers familiar with CLI tools

1. **Security Benefits**

   - Permissions automatically set to `600` (owner read/write only)
   - Outside of project directory (no accidental commits)
   - Protected from other users on system

1. **Convenience**

   - Share tokens across multiple QTrader projects/clones
   - No need to re-authenticate when switching projects
   - Survives project directory deletion

1. **Git-Friendly**

   - Never in project tree
   - No need to worry about `.gitignore` patterns
   - Clean separation of code vs credentials

______________________________________________________________________

## ❌ Not Recommended: Project-Level Storage

**Location:** `/home/username/Projects/QTrader/.tokens/schwab_tokens.json`

### Problems

1. **Duplication** - Need separate tokens for each project clone
1. **Risk** - Easy to accidentally commit if `.gitignore` misconfigured
1. **Maintenance** - Must re-authenticate every time you clone/move project
1. **Confusion** - Unclear which tokens are current if you have multiple clones
1. **Non-Standard** - Doesn't follow common patterns (`.ssh`, `.aws`, etc.)

______________________________________________________________________

## 🐛 Bug Fixed: Literal `~` Directory

### The Issue

The config previously had:

```yaml
token_cache_path: "~/.qtrader/schwab_tokens.json"
```

**Problem:** YAML strings don't expand `~` - it's treated literally!

**Result:** Created `/home/javier/Projects/QTrader/~/.qtrader/` directory (note the literal `~`)

### The Fix

Changed to:

```yaml
token_cache_path: null  # Uses default: ~/.qtrader/schwab_tokens.json
```

**Why:** Let the Python code handle path expansion using `Path.home()`:

```python
if token_cache_path is None:
    token_cache_path = Path.home() / ".qtrader" / "schwab_tokens.json"
```

______________________________________________________________________

## 📋 Implementation Details

### Code Default (Correct)

```python
# In src/qtrader/auth/schwab_oauth.py
if token_cache_path is None:
    token_cache_path = Path.home() / ".qtrader" / "schwab_tokens.json"
```

This correctly resolves to:

- Linux/Mac: `/home/username/.qtrader/schwab_tokens.json`
- Windows: `C:\Users\username\.qtrader\schwab_tokens.json`

### File Permissions

```python
TOKEN_CACHE_FILE.chmod(0o600)  # Owner read/write only
```

Prevents other users on the system from reading your tokens.

### Directory Creation

```python
self.token_cache_path.parent.mkdir(parents=True, exist_ok=True)
```

Automatically creates `~/.qtrader/` if it doesn't exist.

______________________________________________________________________

## 🔒 Security Considerations

### What's Protected

✅ **File Permissions** - `600` (owner only) ✅ **Outside Git** - Never in project directory ✅ **Automatic Cleanup** - Tokens have expiration

### What You Should Do

1. **Never commit tokens** - Already handled by location
1. **Never share tokens** - Personal to your Schwab account
1. **Keep credentials secret** - `SCHWAB_API_KEY` and `SCHWAB_API_SECRET` in env vars only
1. **Rotate regularly** - Tokens auto-refresh (1800s expiry)

### Environment Variables

Store credentials in environment or secure secret manager:

```bash
export SCHWAB_API_KEY="your_key"
export SCHWAB_API_SECRET="your_secret"
```

**Never** in:

- ❌ Code files
- ❌ Config files committed to git
- ❌ Slack/email/docs

______________________________________________________________________

## 📁 Complete Storage Layout

```
/home/username/
├── .qtrader/                           # User-level QTrader config
│   ├── schwab_tokens.json             # OAuth tokens (chmod 600)
│   ├── ssl/                           # SSL certificates
│   │   ├── localhost.pem              # Self-signed cert for OAuth callback
│   │   └── localhost-key.pem          # Private key
│   ├── config.yaml                    # Optional: user-level settings
│   └── data_sources.yaml              # Optional: user-level data sources
│
├── Projects/
│   └── QTrader/                       # Project directory
│       ├── config/                    # Project-level configs (committed)
│       │   ├── qtrader.yaml
│       │   └── data_sources.yaml
│       ├── data/                      # Ignored by git
│       └── output/                    # Ignored by git
```

**Key Point:** User credentials (`~/.qtrader/`) are **separate** from project files.

______________________________________________________________________

## 🔧 Migration (Already Done)

For anyone who had the bug:

```bash
# Move tokens from buggy location
mkdir -p ~/.qtrader
mv ~/Projects/QTrader/~/.qtrader/schwab_tokens.json ~/.qtrader/
chmod 600 ~/.qtrader/schwab_tokens.json

# Remove buggy directory
rm -rf ~/Projects/QTrader/~
```

This has been completed for your installation.

______________________________________________________________________

## 📚 Related Patterns

Similar tools that follow this pattern:

| Tool       | Config Location | Credentials              |
| ---------- | --------------- | ------------------------ |
| SSH        | `~/.ssh/`       | Keys, known_hosts        |
| AWS CLI    | `~/.aws/`       | credentials, config      |
| Docker     | `~/.docker/`    | config.json              |
| Kubernetes | `~/.kube/`      | config                   |
| Git        | `~/.gitconfig`  | User settings            |
| QTrader    | `~/.qtrader/`   | schwab_tokens.json, ssl/ |

**Consistency matters** - developers expect personal credentials in home directory.

______________________________________________________________________

## ✅ Summary

**Best Practice: `/home/username/.qtrader/schwab_tokens.json`**

- User-level storage (not project-level)
- Follows industry conventions
- Secure by default
- Convenient for multiple projects
- Already implemented correctly in code

**Fixed:** YAML config now uses `null` to trigger proper default path expansion.

**Status:** Your tokens are now in the correct location!
