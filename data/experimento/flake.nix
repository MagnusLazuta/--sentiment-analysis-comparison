{
  description = "PLN TF: tools for the inference job (Python itself comes from uv)";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
  outputs = { self, nixpkgs }:
    let pkgs = nixpkgs.legacyPackages.x86_64-linux;
    in {
      devShells.x86_64-linux.default = pkgs.mkShell {
        # uv installs a standalone CPython and manylinux wheels (torch with CUDA);
        # nix only provides the tools, so the system CUDA driver is used as-is.
        packages = [ pkgs.uv pkgs.libarchive pkgs.git pkgs.curl ];
      };
    };
}
