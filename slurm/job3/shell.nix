#
# $ nix-shell --pure --arg withBoost false --argstr tag 20.09
#

{
  tag ? "20.09",
  pymksVersion ? "cf653e004848c9c68ca31a85add0d1ac8611a93f"
}:
let
  file_path = builtins.toString ../.;
#  pkgs = import (builtins.fetchTarball "file://${file_path}/${tag}.tar.gz") {};
  pkgs = import (builtins.fetchTarball "file://${file_path}/nixpkgs-21.11.tar.gz") {};  
  pymkssrc = builtins.fetchTarball "file://${file_path}/pymks-${pymksVersion}.tar.gz";
  pymks = pypkgs.callPackage "${pymkssrc}/default.nix" { graspi = null; };
  pypkgs = pkgs.python3Packages;
  extra = with pypkgs; [ black pylint flake8 zarr pymks h5py memory_profiler click ];
in
  (pymks.overridePythonAttrs (old: rec {

    propagatedBuildInputs = old.propagatedBuildInputs;

    nativeBuildInputs = propagatedBuildInputs ++ extra;

    postShellHook = ''
      export OMPI_MCA_plm_rsh_agent=${pkgs.openssh}/bin/ssh

      SOURCE_DATE_EPOCH=$(date +%s)
      export PYTHONUSERBASE=$PWD/.local
      export USER_SITE=`python -c "import site; print(site.USER_SITE)"`
      export PYTHONPATH=$PYTHONPATH:$USER_SITE
      export PATH=$PATH:$PYTHONUSERBASE/bin

    '';
  }))
