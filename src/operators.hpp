#pragma once

#include <utility>
#include <cassert>

// Note that this has to be updated carefully. They are ordered by arity.
// Furthermore note that op_code::LOG is currently also used to distinguish between arities (so add other unary operators before LOG).
enum class op_code : int {
    SIN, 
    COS,
    SQRT,
    EXP,
    LOG,

    ADD,
    SUB,
    MUL,
    DIV,
    
    LENGTH 
};

// Note that this has to be updated carefully.
// The current implementation relies on arity = static_cast<int>(OP_n) + 1.
// Also, x <= static_cast<int>(OP_2) is also used as a seperator between "operators" and constants
enum class type : int {
    OP_1,           // unary operation
    OP_2,           // binary operation
    VARIABLE_INDEX,  
    CONSTANT 
};

inline std::pair<int, type> opcode_arity(int const op) {
    assert(op < static_cast<int>(op_code::LENGTH));

    return op <= static_cast<int>(op_code::LOG)
        ? std::pair{1, type::OP_1}
        : std::pair{2, type::OP_2}; 
}

inline std::pair<int, type> opcode_arity(op_code const op) {
    return opcode_arity(static_cast<int>(op));
}

