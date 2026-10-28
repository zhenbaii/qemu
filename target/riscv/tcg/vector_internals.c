/*
 * RISC-V Vector Extension Internals
 *
 * Copyright (c) 2020 T-Head Semiconductor Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms and conditions of the GNU General Public License,
 * version 2 or later, as published by the Free Software Foundation.
 *
 * This program is distributed in the hope it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "qemu/osdep.h"
#include "vector_internals.h"

bool riscv_decode_vtype(const RISCVCPUConfig *cfg, int xlen,
                        uint64_t vtype, RISCVVType *decoded)
{
    uint32_t vlmul = FIELD_EX64(vtype, VTYPE, VLMUL);
    uint32_t vsew = FIELD_EX64(vtype, VTYPE, VSEW);
    uint32_t sew = 8 << vsew;
    bool altfmt = FIELD_EX64(vtype, VTYPE, ALTFMT);
    bool ill_altfmt = true;
    bool vill = (vtype >> (xlen - 1)) & 1;
    uint64_t reserved;

    if (vlmul & 4) {
        vill |= vlmul == 4 || (cfg->elen >> (8 - vlmul)) < sew;
    }

    if (vsew == MO_8 || vsew == MO_16) {
        ill_altfmt = !cfg->ext_zvfbfa;
    }
    vill |= altfmt && ill_altfmt;

    if (cfg->ext_zvfbfa) {
        reserved = vtype & MAKE_64BIT_MASK(R_VTYPE_RESERVED_SHIFT,
                                           xlen - 1 - R_VTYPE_RESERVED_SHIFT);
    } else {
        reserved = vtype & MAKE_64BIT_MASK(R_VTYPE_ALTFMT_SHIFT,
                                           xlen - 1 - R_VTYPE_ALTFMT_SHIFT);
    }
    vill |= sew > cfg->elen || reserved != 0;
    if (vill) {
        return false;
    }

    decoded->vsew = vsew;
    decoded->lmul = sextract32(vlmul, 0, 3);
    decoded->vta = FIELD_EX64(vtype, VTYPE, VTA);
    decoded->vma = FIELD_EX64(vtype, VTYPE, VMA);
    decoded->altfmt = altfmt;
    decoded->vlmax = vext_get_vlmax(cfg->vlenb, vsew, decoded->lmul);
    return true;
}

/* set agnostic elements to 1s */
void vext_set_elems_1s(void *base, uint32_t is_agnostic, uint32_t cnt,
                       uint32_t tot)
{
    if (is_agnostic == 0) {
        /* policy undisturbed */
        return;
    }
    if (tot - cnt == 0) {
        return ;
    }

    if (HOST_BIG_ENDIAN) {
        /*
         * Deal the situation when the elements are insdie
         * only one uint64 block including setting the
         * masked-off element.
         */
        if (((tot - 1) ^ cnt) < 8) {
            memset(base + H1(tot - 1), -1, tot - cnt);
            return;
        }
        /*
         * Otherwise, at least cross two uint64_t blocks.
         * Set first unaligned block.
         */
        if (cnt % 8 != 0) {
            uint32_t j = ROUND_UP(cnt, 8);
            memset(base + H1(j - 1), -1, j - cnt);
            cnt = j;
        }
        /* Set other 64bit aligend blocks */
    }
    memset(base + cnt, -1, tot - cnt);
}

void do_vext_vv(void *vd, void *v0, void *vs1, void *vs2,
                CPURISCVState *env, uint32_t desc,
                opivv2_fn *fn, uint32_t esz)
{
    uint32_t vm = vext_vm(desc);
    uint32_t vl = env->vl;
    uint32_t total_elems = vext_get_total_elems(env, desc, esz);
    uint32_t vta = vext_vta(desc);
    uint32_t vma = vext_vma(desc);
    uint32_t i;

    VSTART_CHECK_EARLY_EXIT(env, vl);

    for (i = env->vstart; i < vl; i++) {
        if (!vm && !vext_elem_mask(v0, i)) {
            /* set masked-off elements to 1s */
            vext_set_elems_1s(vd, vma, i * esz, (i + 1) * esz);
            continue;
        }
        fn(vd, vs1, vs2, i);
    }
    env->vstart = 0;
    /* set tail elements to 1s */
    vext_set_elems_1s(vd, vta, vl * esz, total_elems * esz);
}

void do_vext_vx(void *vd, void *v0, target_long s1, void *vs2,
                CPURISCVState *env, uint32_t desc,
                opivx2_fn fn, uint32_t esz)
{
    uint32_t vm = vext_vm(desc);
    uint32_t vl = env->vl;
    uint32_t total_elems = vext_get_total_elems(env, desc, esz);
    uint32_t vta = vext_vta(desc);
    uint32_t vma = vext_vma(desc);
    uint32_t i;

    VSTART_CHECK_EARLY_EXIT(env, vl);

    for (i = env->vstart; i < vl; i++) {
        if (!vm && !vext_elem_mask(v0, i)) {
            /* set masked-off elements to 1s */
            vext_set_elems_1s(vd, vma, i * esz, (i + 1) * esz);
            continue;
        }
        fn(vd, s1, vs2, i);
    }
    env->vstart = 0;
    /* set tail elements to 1s */
    vext_set_elems_1s(vd, vta, vl * esz, total_elems * esz);
}
