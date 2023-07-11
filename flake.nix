{
  description = "twesterhout/annealing-sign-problem: Unveiling ground state sign structures of frustrated quantum systems via non-glassy Ising models";

  nixConfig = {
    extra-experimental-features = "nix-command flakes";
  };

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    nix-filter.url = "github:numtide/nix-filter";
    flake-compat = {
      url = "github:edolstra/flake-compat";
      # don't look for a flake.nix file in this repository
      # this tells Nix to retrieve this input as just source code
      flake = false;
    };
  };

  outputs = inputs: inputs.flake-utils.lib.eachDefaultSystem (system:
    with builtins;
    let
      inherit (inputs.nixpkgs) lib;
      pkgs = import inputs.nixpkgs { inherit system; };

      my-python = pkgs.python3.withPackages (ps: with ps; [
        cffi
        loguru
        h5py
        numpy
        pyyaml
        scipy
      ]);
    in
    {
      devShells.default =
        pkgs.mkShell {
          packages = [ ];
          nativeBuildInputs = with pkgs; [
            my-python
            # lsp support for Python
            python3Packages.black
            nodePackages.pyright
            # plotting etc.
            gnuplot_qt
            imagemagick
          ];
          shellHook = ''
            export PROMPT_COMMAND=""
            export PS1='🐍 Python ${pkgs.python3.version} \w $ '
          '';
        };
      formatter = pkgs.nixpkgs-fmt;
    });
}
