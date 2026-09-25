#!/usr/bin/env bash
# Standalone release bootstrap. Keep pins synchronized with release/bootstrap-tools.json.
set -euo pipefail
usage() {
  cat <<'EOF'
Usage: ./install.sh (--cpu | --cuda cu126) [--version X.Y.Z] [--prefix PATH] [--upgrade]
Install a published Sym binary release into a private Python environment.
Default prefix: ${XDG_DATA_HOME:-$HOME/.local/share}/sym
Requires Linux x86_64, glibc >= 2.28, Bash, curl, tar, sha256sum.
CUDA installation additionally requires a working compatible NVIDIA driver/GPU.
EOF
}
fail() { printf 'Sym installation failed: %s\n' "$*" >&2; exit 1; }
sanitize_uv_environment() {
  local name
  for name in "${!UV_@}" "${!PIP_@}"; do
    [[ -z "$name" ]] || unset "$name"
  done
  unset PYTHONHOME PYTHONPATH VIRTUAL_ENV
}
main() {
variant=''
version=''
prefix="${XDG_DATA_HOME:-$HOME/.local/share}/sym"
upgrade=()
while (($#)); do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --cpu) [[ -z "$variant" ]] || fail 'select exactly one variant'; variant=cpu; shift ;;
    --cuda) [[ -z "$variant" && $# -ge 2 && "$2" == cu126 ]] || fail 'use --cuda cu126'; variant=cu126; shift 2 ;;
    --version) [[ $# -ge 2 && "$2" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || fail 'invalid --version'; version="$2"; shift 2 ;;
    --prefix) [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || fail 'missing --prefix'; prefix="$2"; shift 2 ;;
    --upgrade) upgrade=(--upgrade); shift ;;
    *) fail "unknown argument: $1" ;;
  esac
done
[[ -n "$variant" ]] || fail 'select --cpu or --cuda cu126'
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || fail 'Linux x86_64 required'
for cmd in curl tar sha256sum mktemp getconf; do command -v "$cmd" >/dev/null || fail "missing prerequisite: $cmd"; done
getconf GNU_LIBC_VERSION >/dev/null 2>&1 || fail 'glibc Linux required'
[[ ! -L "$prefix" ]] || fail 'prefix must not be a symlink'
mkdir -p "$prefix"
prefix="$(cd "$prefix" && pwd -P)"
for entry in python cache; do
  [[ ! -L "$prefix/$entry" ]] || fail "$entry must not be a symlink"
  mkdir -p "$prefix/$entry"
done
bootstrap="$(mktemp -d "$prefix/.bootstrap-XXXXXXXX")"
trap 'rm -rf -- "$bootstrap"' EXIT
curl_args=(--fail --show-error --silent --location --proto '=https' --proto-redir '=https' --retry 2 --connect-timeout 30)
# Release manifest absence is reported before the large interpreter download.
base=https://github.com/JueonPark/sym/releases
if [[ -n "$version" ]]; then manifest_url="$base/download/v$version/installation-manifest.json";
else manifest_url="$base/latest/download/installation-manifest.json"; fi
curl "${curl_args[@]}" "$manifest_url" -o "$bootstrap/manifest.json" || fail 'no installation manifest is available for this release; see docs/installation.md for the source-build route'
curl "${curl_args[@]}" https://github.com/astral-sh/uv/releases/download/0.12.11/uv-x86_64-unknown-linux-gnu.tar.gz -o "$bootstrap/uv.tar.gz"
printf '%s  %s\n' 4ae93e0f148a18434cc094072547cec88912fc4a72b984183c7d0d0e9586cb5e "$bootstrap/uv.tar.gz" | sha256sum --check --status || fail 'uv checksum mismatch'
# Extract one known file from a verified upstream archive, never arbitrary paths.
tar -xzf "$bootstrap/uv.tar.gz" -C "$bootstrap" uv-x86_64-unknown-linux-gnu/uv
uv="$bootstrap/uv-x86_64-unknown-linux-gnu/uv"
sanitize_uv_environment
export UV_NO_CONFIG=1 UV_CACHE_DIR="$prefix/cache" UV_PYTHON_INSTALL_DIR="$prefix/python"
"$uv" python install cpython-3.14.7-linux-x86_64-gnu --no-bin
python="$prefix/python/cpython-3.14.7-linux-x86_64-gnu/bin/python3.14"
[[ -x "$python" ]] || fail 'uv did not supply the qualified Python interpreter'
# Verify helper bytes before executing them. Full admission is repeated by the helper.
"$python" -I - "$bootstrap" "$variant" "$version" <<'PY'
import hashlib,json,pathlib,re,sys,urllib.request
from urllib.parse import urlsplit,unquote
root=pathlib.Path(sys.argv[1]); variant=sys.argv[2]; requested=sys.argv[3]
m=json.loads((root/'manifest.json').read_text())
if any(type(m.get(k)) is not int or m[k]!=1 for k in ('schema_version','installer_protocol')):
    raise SystemExit('unsupported installation manifest schema/protocol')
version=m.get('release_version','')
if not re.fullmatch(r'(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)',version) or (requested and requested!=version):
    raise SystemExit('release version mismatch')
try: asset=m['variants'][variant]['assets']['helper']
except (KeyError,TypeError): raise SystemExit('requested variant/helper is absent')
name=asset.get('filename',''); url=asset.get('url','')
if not name or '/' in name or '\\' in name or name in ('.','..'):
    raise SystemExit('invalid helper filename')
if not url.startswith(f'https://github.com/JueonPark/sym/releases/download/v{version}/') or unquote(urlsplit(url).path.rsplit('/',1)[-1])!=name:
    raise SystemExit('invalid helper URL')
if type(asset.get('size')) is not int or not 0<asset['size']<=10485760 or not re.fullmatch('[0-9a-f]{64}',asset.get('sha256','')):
    raise SystemExit('invalid helper size/hash')
class Redirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        p=urlsplit(newurl)
        if p.scheme!='https' or p.hostname not in ('github.com','release-assets.githubusercontent.com','objects.githubusercontent.com'):
            raise SystemExit('unsupported helper redirect')
        return super().redirect_request(req,fp,code,msg,headers,newurl)
with urllib.request.build_opener(Redirects()).open(url,timeout=60) as response:
    payload=response.read(asset['size']+1)
if len(payload)!=asset['size'] or hashlib.sha256(payload).hexdigest()!=asset['sha256']:
    raise SystemExit('helper size/checksum mismatch')
(root/'installer.pyz').write_bytes(payload)
PY
"$python" -I "$bootstrap/installer.pyz" --manifest "$bootstrap/manifest.json" --prefix "$prefix" --variant "$variant" --uv "$uv" "${upgrade[@]}"

}
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then main "$@"; fi
