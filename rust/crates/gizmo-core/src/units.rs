macro_rules! code_unit {
    ($name:ident) => {
        #[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
        #[repr(transparent)]
        pub struct $name(f64);

        impl $name {
            #[must_use]
            pub const fn new(value: f64) -> Self {
                Self(value)
            }

            #[must_use]
            pub const fn value(self) -> f64 {
                self.0
            }

            #[must_use]
            pub fn is_finite(self) -> bool {
                self.0.is_finite()
            }
        }
    };
}

code_unit!(CodeLength);
code_unit!(CodeMass);
code_unit!(CodeTime);
code_unit!(CodeVelocity);
