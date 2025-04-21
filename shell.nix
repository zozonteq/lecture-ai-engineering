{
  pkgs ? import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/refs/tags/24.11.zip") { },
}:
pkgs.mkShell {
  buildInputs = with pkgs; [
    python311Full
    direnv
  ];
}