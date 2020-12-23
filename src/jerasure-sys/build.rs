extern crate cc;

fn main() {
    cc::Build::new()
        .files(&[
            "jerasure/src/galois.c",
            "jerasure/src/jerasure.c",
            "jerasure/src/reed_sol.c",
            "jerasure/src/cauchy.c",
            "jerasure/src/liberation.c",
        ])
        .include("jerasure/include")
        .include("gf-complete/include")
        .compile("Jerasure");
    println!("cargo:rustc-link-lib=static=Jerasure");
}
