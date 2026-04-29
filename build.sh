#!/usr/bin/env bash
set -euxo pipefail

cd "$(dirname "$0")"

# cli/package.json has a "prepare" hook that runs "build" during npm install,
# but the build needs generated proto files that don't exist yet.
# strip it before install, restore after.
node -e "
const pkg = require('./cli/package.json');
delete pkg.scripts.prepare;
require('fs').writeFileSync('./cli/package.json', JSON.stringify(pkg, null, '\t') + '\n');
"
trap 'git checkout cli/package.json 2>/dev/null' EXIT

npm install
cd webview-ui && npm install && cd ..
npm run protos
cd cli
npx tsx esbuild.mts
npm link
echo "done — dirac now points to local build (undo with: npm unlink -g dirac-cli)"
