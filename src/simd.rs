#[derive(Default, Clone, Debug, Copy)]
#[repr(C, align(32))]
pub struct F32x16 {
    pub r0: f32,
    pub r1: f32,
    pub r2: f32,
    pub r3: f32,
    pub r4: f32,
    pub r5: f32,
    pub r6: f32,
    pub r7: f32,
    pub r8: f32,
    pub r9: f32,
    pub r10: f32,
    pub r11: f32,
    pub r12: f32,
    pub r13: f32,
    pub r14: f32,
    pub r15: f32,
}

impl F32x16 {
    pub fn new(acc: &[f32; SIMD_LANES_SIZE]) -> Self {
        Self {
            r0: acc[0],
            r1: acc[1],
            r2: acc[2],
            r3: acc[3],
            r4: acc[4],
            r5: acc[5],
            r6: acc[6],
            r7: acc[7],
            r8: acc[8],
            r9: acc[9],
            r10: acc[10],
            r11: acc[11],
            r12: acc[12],
            r13: acc[13],
            r14: acc[14],
            r15: acc[15],
        }
    }

    pub fn almost_same(&self, rhs: &Self) -> bool {
        (self.r0 - rhs.r0).abs()
            + (self.r1 - rhs.r1).abs()
            + (self.r2 - rhs.r2).abs()
            + (self.r3 - rhs.r3).abs()
            + (self.r4 - rhs.r4).abs()
            + (self.r5 - rhs.r5).abs()
            + (self.r6 - rhs.r6).abs()
            + (self.r7 - rhs.r7).abs()
            + (self.r8 - rhs.r8).abs()
            + (self.r9 - rhs.r9).abs()
            + (self.r10 - rhs.r10).abs()
            + (self.r11 - rhs.r11).abs()
            + (self.r12 - rhs.r12).abs()
            + (self.r13 - rhs.r13).abs()
            + (self.r14 - rhs.r14).abs()
            + (self.r15 - rhs.r15).abs()
            < crate::EPS
    }

    pub fn sum(&self) -> f32 {
        self.r0
            + self.r1
            + self.r2
            + self.r3
            + self.r4
            + self.r5
            + self.r6
            + self.r7
            + self.r8
            + self.r9
            + self.r10
            + self.r11
            + self.r12
            + self.r13
            + self.r14
            + self.r15
    }

    pub fn mul(self, rhs: &Self) -> Self {
        Self {
            r0: self.r0 * rhs.r0,
            r1: self.r1 * rhs.r1,
            r2: self.r2 * rhs.r2,
            r3: self.r3 * rhs.r3,
            r4: self.r4 * rhs.r4,
            r5: self.r5 * rhs.r5,
            r6: self.r6 * rhs.r6,
            r7: self.r7 * rhs.r7,
            r8: self.r8 * rhs.r8,
            r9: self.r9 * rhs.r9,
            r10: self.r10 * rhs.r10,
            r11: self.r11 * rhs.r11,
            r12: self.r12 * rhs.r12,
            r13: self.r13 * rhs.r13,
            r14: self.r14 * rhs.r14,
            r15: self.r15 * rhs.r15,
        }
    }

    pub fn mul_assign(&mut self, rhs: Self) {
        self.r0 *= rhs.r0;
        self.r1 *= rhs.r1;
        self.r2 *= rhs.r2;
        self.r3 *= rhs.r3;
        self.r4 *= rhs.r4;
        self.r5 *= rhs.r5;
        self.r6 *= rhs.r6;
        self.r7 *= rhs.r7;
        self.r8 *= rhs.r8;
        self.r9 *= rhs.r9;
        self.r10 *= rhs.r10;
        self.r11 *= rhs.r11;
        self.r12 *= rhs.r12;
        self.r13 *= rhs.r13;
        self.r14 *= rhs.r14;
        self.r15 *= rhs.r15;
    }

    pub fn sub_assign(&mut self, rhs: Self) {
        self.r0 -= rhs.r0;
        self.r1 -= rhs.r1;
        self.r2 -= rhs.r2;
        self.r3 -= rhs.r3;
        self.r4 -= rhs.r4;
        self.r5 -= rhs.r5;
        self.r6 -= rhs.r6;
        self.r7 -= rhs.r7;
        self.r8 -= rhs.r8;
        self.r9 -= rhs.r9;
        self.r10 -= rhs.r10;
        self.r11 -= rhs.r11;
        self.r12 -= rhs.r12;
        self.r13 -= rhs.r13;
        self.r14 -= rhs.r14;
        self.r15 -= rhs.r15;
    }
}

pub const SIMD_LANES_SIZE: usize = 16;
