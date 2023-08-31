import json
import random
from sortedcontainers import SortedSet
from operator import neg
from values.constraints4x4 import constraints
from values.constraints9x9 import sudoku_constraints
from flask import Flask

with open('values/9x9variables.txt') as f:
    variablesCSP = f.read()

#replace with puzzle to be solved, note 4x4domains is the given puzzle for the 4x4, sudoku1 is first puzzle
#currently set for 9x9
with open('values/sudoku1.txt') as f:
    domainsCSP = f.read()
    

domainsCSP = json.loads(domainsCSP)
variablesCSP = variablesCSP.split(",")
constraintsCSP = sudoku_constraints

#print(domainsCSP)
print(type(constraintsCSP))
#print(variablesCSP)

#print(variablesCSP)

#Functions used from each file marked
#https://github.com/aimacode/aima-python/blob/master/csp.py is aimacsp
#https://github.com/aimacode/aima-python/blob/master/utils.py is aimautil

#adapted from aimacsp
#class for all things csp related 
class CSP:
    def __init__(self, variables, domains, neighbors, constraints):
        variables = variables or list(domains.keys())
        self.variables = variables
        self.domains = domains
        self.neighbors = neighbors
        self.constraints = constraints
        
    def suppose(self, var, value):
        """Start accumulating inferences from assuming var=value."""
        removals = [(var, a) for a in self.domains[var] if a != value]
        self.domains[var] = [value]
        return removals

    def restore(self, removals):
        """Undo a supposition and all inferences from it."""
        for a, b in removals:
            self.domains[a].append(b)
    
    def unassign(self, var, assignment):
        """Remove var from assignment (if it is there)"""
        if var in assignment:
            assignment[var] = None

#adapted from aimacsp
def order_domain_values(var, assignment, csp):
    return sorted(csp.domains[var])

#adapted from aimacsp
#
def revise(csp, Xi, Xj):
    #assume not remove value
    revised = False
    for a in csp.domains[Xi][:]:
        if (not any(a != b for b in csp.domains[Xj])):
            #remove value
            index = csp.domains[Xi].index(a)
            csp.domains[Xi].remove(csp.domains[Xi][index])
            revised = True
        else:
            revised = False
        #print(a)
    return revised

#adapted from aimacsp
def AC_3(csp):
    queue = {(Xi, Xk) for Xi in csp.variables for Xk in csp.neighbors[Xi]}
    #print(queue)
    while queue:
        (Xi, Xj) = queue.pop()
        #print(Xi, Xj)
        #print((Xi, Xj))
        revised = revise(csp, Xi, Xj)
        #print(revised)
        if revised:
            if (not csp.domains[Xi]):
                return False
                print("AAAAAAAAAAAAAA")
            for Xk in csp.neighbors[Xi]:
                if Xk != Xj:
                    queue.add((Xk, Xi))
                    print("BBBBBBBBBBB")

    return True 

#adapted from aimacsp
def minimum_remaining_values(csp, assigmentset):
    return min([v for v in csp.variables if v not in assigmentset],
                             key=lambda var: len(csp.domains[var]))


#adapted from aimacsp
def backtracking_search(csp):
    def backtrack(assignment):
        #print(assignment)
        #print(len(assignment))
        if (len(assignment) == len(csp.variables)):
            return assignment
        var = minimum_remaining_values(csp, assignment)
        #print(assignment)
        for value in order_domain_values(var, assignment, csp):
            assignment[var] = value
            removals = csp.suppose(var, value)
            if  AC_3(csp):
                return backtrack(assignment)
            csp.restore(removals)
        csp.unassign(var, assignment)
        #print(csp)
    return_result = backtrack({})
    print(return_result)
    return return_result

#4x4neighbors
neighborsCSP4x4 = {'C11':['C12', 'C13', 'C14', 'C21', 'C31', 'C41', 'C22'], 'C12':['C11', 'C13', 'C14', 'C22', 'C32', 'C42', 'C21'], 'C13':['C11', 'C12', 'C14', 'C23', 'C33', 'C43', 'C24'],
                'C14':['C11', 'C12', 'C13', 'C24', 'C34', 'C44', 'C23'], 'C21':['C11', 'C31', 'C41', 'C22', 'C23', 'C24', 'C12'], 'C22':['C21', 'C23', 'C24', 'C12', 'C32', 'C42', 'C11'],
                'C23':['C21', 'C22', 'C24', 'C13', 'C33', 'C43', 'C14'], 'C24':['C21', 'C22', 'C23', 'C14', 'C34', 'C44', 'C13'], 'C31':['C32', 'C33', 'C34', 'C11', 'C12', 'C41', 'C42'],
                'C32':['C31', 'C33', 'C34', 'C12', 'C22', 'C42', 'C41'], 'C33':['C31', 'C32', 'C34', 'C13', 'C23', 'C43', 'C44'], 'C34':['C31', 'C32', 'C33', 'C14', 'C24', 'C44', 'C43'],
                'C41':['C11', 'C21', 'C31', 'C42', 'C43', 'C44', 'C32'], 'C42':['C41', 'C43', 'C44', 'C12', 'C22', 'C32', 'C31'], 'C43':['C41', 'C42', 'C44', 'C13', 'C23', 'C33', 'C34'],
                'C44':['C41', 'C42', 'C43', 'C14', 'C24', 'C34', 'C33']}

#9x9neighbors
neighborsCSP9x9 = {'C11':['C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C21', 'C31', 'C41', 'C51', 'C61', 'C71', 'C81', 'C91', 'C22', 'C23', 'C32', 'C33'],
                'C12':['C11', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C22', 'C32', 'C42', 'C52', 'C62', 'C72', 'C82', 'C92', 'C21', 'C23', 'C31', 'C33'],
                'C13':['C12', 'C11', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C23', 'C33', 'C43', 'C53', 'C63', 'C73', 'C83', 'C93', 'C21', 'C22', 'C31', 'C32'],
                'C14':['C12', 'C13', 'C11', 'C15', 'C16', 'C17', 'C18', 'C19', 'C24', 'C34', 'C44', 'C54', 'C64', 'C74', 'C84', 'C94', 'C25', 'C26', 'C35', 'C36'],
                'C15':['C12', 'C13', 'C11', 'C14', 'C16', 'C17', 'C18', 'C19', 'C25', 'C35', 'C45', 'C55', 'C65', 'C75', 'C85', 'C95', 'C24', 'C26', 'C34', 'C36'],
                'C16':['C12', 'C13', 'C11', 'C15', 'C14', 'C17', 'C18', 'C19', 'C26', 'C36', 'C46', 'C56', 'C66', 'C76', 'C86', 'C96', 'C24', 'C25', 'C34', 'C35'],
                'C17':['C12', 'C13', 'C11', 'C15', 'C16', 'C14', 'C18', 'C19', 'C27', 'C37', 'C47', 'C57', 'C67', 'C77', 'C87', 'C97', 'C28', 'C29', 'C38', 'C39'],
                'C18':['C12', 'C13', 'C11', 'C15', 'C16', 'C17', 'C14', 'C19', 'C28', 'C38', 'C48', 'C58', 'C68', 'C78', 'C88', 'C98', 'C27', 'C29', 'C37', 'C39'],
                'C19':['C12', 'C13', 'C11', 'C15', 'C16', 'C17', 'C18', 'C14', 'C29', 'C39', 'C49', 'C59', 'C69', 'C79', 'C89', 'C99', 'C27', 'C28', 'C37', 'C38'],
                'C21':['C22', 'C23', 'C24', 'C25', 'C26', 'C27', 'C28', 'C29', 'C11', 'C31', 'C41', 'C51', 'C61', 'C71', 'C81', 'C91', 'C12', 'C13', 'C32', 'C33'],
                'C22':['C21', 'C23', 'C24', 'C25', 'C26', 'C27', 'C28', 'C29', 'C12', 'C32', 'C42', 'C52', 'C62', 'C72', 'C82', 'C92', 'C11', 'C13', 'C31', 'C33'],
                'C23':['C22', 'C21', 'C24', 'C25', 'C26', 'C27', 'C28', 'C29', 'C13', 'C33', 'C43', 'C53', 'C63', 'C73', 'C83', 'C93', 'C11', 'C12', 'C31', 'C32'],
                'C24':['C22', 'C23', 'C21', 'C25', 'C26', 'C27', 'C28', 'C29', 'C14', 'C34', 'C44', 'C54', 'C64', 'C74', 'C84', 'C94', 'C15', 'C16', 'C35', 'C36'],
                'C25':['C22', 'C23', 'C24', 'C21', 'C26', 'C27', 'C28', 'C29', 'C15', 'C35', 'C45', 'C55', 'C65', 'C75', 'C85', 'C95', 'C14', 'C16', 'C34', 'C36'],
                'C26':['C22', 'C23', 'C24', 'C25', 'C21', 'C27', 'C28', 'C29', 'C16', 'C36', 'C46', 'C56', 'C66', 'C76', 'C86', 'C96', 'C14', 'C15', 'C34', 'C35'],
                'C27':['C22', 'C23', 'C24', 'C25', 'C26', 'C21', 'C28', 'C29', 'C17', 'C37', 'C47', 'C57', 'C67', 'C77', 'C87', 'C97', 'C18', 'C19', 'C38', 'C39'],
                'C28':['C22', 'C23', 'C24', 'C25', 'C26', 'C27', 'C21', 'C29', 'C18', 'C38', 'C48', 'C58', 'C68', 'C78', 'C88', 'C98', 'C17', 'C19', 'C37', 'C39'],
                'C29':['C22', 'C23', 'C24', 'C25', 'C26', 'C27', 'C28', 'C21', 'C19', 'C39', 'C49', 'C59', 'C69', 'C79', 'C89', 'C99', 'C17', 'C18', 'C37', 'C38'],
                'C31':['C32', 'C33', 'C34', 'C35', 'C36', 'C37', 'C38', 'C39', 'C11', 'C21', 'C41', 'C51', 'C61', 'C71', 'C81', 'C91', 'C12', 'C13', 'C22', 'C23'],
                'C32':['C31', 'C33', 'C34', 'C35', 'C36', 'C37', 'C38', 'C39', 'C12', 'C22', 'C42', 'C52', 'C62', 'C72', 'C82', 'C92', 'C11', 'C13', 'C21', 'C23'],
                'C33':['C32', 'C31', 'C34', 'C35', 'C36', 'C37', 'C38', 'C39', 'C13', 'C23', 'C43', 'C53', 'C63', 'C73', 'C83', 'C93', 'C11', 'C12', 'C21', 'C22'],
                'C34':['C32', 'C33', 'C31', 'C35', 'C36', 'C37', 'C38', 'C39', 'C14', 'C24', 'C44', 'C54', 'C64', 'C74', 'C84', 'C94', 'C15', 'C16', 'C25', 'C26'],
                'C35':['C32', 'C33', 'C34', 'C31', 'C36', 'C37', 'C38', 'C39', 'C15', 'C25', 'C45', 'C55', 'C65', 'C75', 'C85', 'C95', 'C14', 'C16', 'C24', 'C26'],
                'C36':['C32', 'C33', 'C34', 'C35', 'C31', 'C37', 'C38', 'C39', 'C16', 'C26', 'C46', 'C56', 'C66', 'C76', 'C86', 'C96', 'C14', 'C15', 'C24', 'C25'],
                'C37':['C32', 'C33', 'C34', 'C35', 'C36', 'C31', 'C38', 'C39', 'C17', 'C27', 'C47', 'C57', 'C67', 'C77', 'C87', 'C97', 'C18', 'C19', 'C28', 'C29'],
                'C38':['C32', 'C33', 'C34', 'C35', 'C36', 'C37', 'C31', 'C39', 'C18', 'C28', 'C48', 'C58', 'C68', 'C78', 'C88', 'C98', 'C17', 'C19', 'C27', 'C29'],
                'C39':['C32', 'C33', 'C34', 'C35', 'C36', 'C37', 'C38', 'C31', 'C19', 'C29', 'C49', 'C59', 'C69', 'C79', 'C89', 'C99', 'C17', 'C18', 'C27', 'C28'],
                'C41':['C42', 'C43', 'C44', 'C45', 'C46', 'C47', 'C48', 'C49', 'C11', 'C21', 'C31', 'C51', 'C61', 'C71', 'C81', 'C91', 'C52', 'C53', 'C62', 'C63'],
                'C42':['C41', 'C43', 'C44', 'C45', 'C46', 'C47', 'C48', 'C49', 'C12', 'C22', 'C32', 'C52', 'C62', 'C72', 'C82', 'C92', 'C51', 'C53', 'C61', 'C63'],
                'C43':['C42', 'C41', 'C44', 'C45', 'C46', 'C47', 'C48', 'C49', 'C13', 'C23', 'C33', 'C53', 'C63', 'C73', 'C83', 'C93', 'C51', 'C52', 'C61', 'C62'],
                'C44':['C42', 'C43', 'C41', 'C45', 'C46', 'C47', 'C48', 'C49', 'C14', 'C24', 'C34', 'C54', 'C64', 'C74', 'C84', 'C94', 'C55', 'C56', 'C65', 'C66'],
                'C45':['C42', 'C43', 'C44', 'C41', 'C46', 'C47', 'C48', 'C49', 'C15', 'C25', 'C35', 'C55', 'C65', 'C75', 'C85', 'C95', 'C54', 'C56', 'C64', 'C66'],
                'C46':['C42', 'C43', 'C44', 'C45', 'C41', 'C47', 'C48', 'C49', 'C16', 'C26', 'C36', 'C56', 'C66', 'C76', 'C86', 'C96', 'C54', 'C55', 'C64', 'C65'],
                'C47':['C42', 'C43', 'C44', 'C45', 'C46', 'C41', 'C48', 'C49', 'C17', 'C27', 'C37', 'C57', 'C67', 'C77', 'C87', 'C97', 'C58', 'C59', 'C68', 'C69'],
                'C48':['C42', 'C43', 'C44', 'C45', 'C46', 'C47', 'C41', 'C49', 'C18', 'C28', 'C38', 'C58', 'C68', 'C78', 'C88', 'C98', 'C57', 'C59', 'C67', 'C69'],
                'C49':['C42', 'C43', 'C44', 'C45', 'C46', 'C47', 'C48', 'C41', 'C19', 'C29', 'C39', 'C59', 'C69', 'C79', 'C89', 'C99', 'C57', 'C58', 'C67', 'C68'],
                'C51':['C52', 'C53', 'C54', 'C55', 'C56', 'C57', 'C58', 'C59', 'C11', 'C31', 'C41', 'C21', 'C61', 'C71', 'C81', 'C91', 'C42', 'C43', 'C62', 'C63'],
                'C52':['C51', 'C53', 'C54', 'C55', 'C56', 'C57', 'C58', 'C59', 'C12', 'C32', 'C42', 'C22', 'C62', 'C72', 'C82', 'C92', 'C41', 'C43', 'C61', 'C63'],
                'C53':['C52', 'C51', 'C54', 'C55', 'C56', 'C57', 'C58', 'C59', 'C13', 'C33', 'C43', 'C23', 'C63', 'C73', 'C83', 'C93', 'C41', 'C42', 'C61', 'C62'],
                'C54':['C52', 'C53', 'C51', 'C55', 'C56', 'C57', 'C58', 'C59', 'C14', 'C34', 'C44', 'C24', 'C64', 'C74', 'C84', 'C94', 'C45', 'C46', 'C65', 'C66'],
                'C55':['C52', 'C53', 'C54', 'C51', 'C56', 'C57', 'C58', 'C59', 'C15', 'C35', 'C45', 'C25', 'C65', 'C75', 'C85', 'C95', 'C44', 'C46', 'C64', 'C66'],
                'C56':['C52', 'C53', 'C54', 'C55', 'C51', 'C57', 'C58', 'C59', 'C16', 'C36', 'C46', 'C26', 'C66', 'C76', 'C86', 'C96', 'C44', 'C45', 'C64', 'C65'],
                'C57':['C52', 'C53', 'C54', 'C55', 'C56', 'C51', 'C58', 'C59', 'C17', 'C37', 'C47', 'C27', 'C67', 'C77', 'C87', 'C97', 'C48', 'C49', 'C68', 'C69'],
                'C58':['C52', 'C53', 'C54', 'C55', 'C56', 'C57', 'C51', 'C59', 'C18', 'C38', 'C48', 'C28', 'C68', 'C78', 'C88', 'C98', 'C47', 'C49', 'C67', 'C69'],
                'C59':['C52', 'C53', 'C54', 'C55', 'C56', 'C57', 'C58', 'C51', 'C19', 'C39', 'C49', 'C29', 'C69', 'C79', 'C89', 'C99', 'C47', 'C48', 'C67', 'C68'],
                'C61':['C62', 'C63', 'C64', 'C65', 'C66', 'C67', 'C68', 'C69', 'C11', 'C21', 'C41', 'C51', 'C31', 'C71', 'C81', 'C91', 'C42', 'C43', 'C52', 'C53'],
                'C62':['C61', 'C63', 'C64', 'C65', 'C66', 'C67', 'C68', 'C69', 'C12', 'C22', 'C42', 'C52', 'C32', 'C72', 'C82', 'C92', 'C41', 'C43', 'C51', 'C53'],
                'C63':['C62', 'C61', 'C64', 'C65', 'C66', 'C67', 'C68', 'C69', 'C13', 'C23', 'C43', 'C53', 'C33', 'C73', 'C83', 'C93', 'C41', 'C42', 'C51', 'C52'],
                'C64':['C62', 'C63', 'C61', 'C65', 'C66', 'C67', 'C68', 'C69', 'C14', 'C24', 'C44', 'C54', 'C34', 'C74', 'C84', 'C94', 'C45', 'C46', 'C55', 'C56'],
                'C65':['C62', 'C63', 'C64', 'C61', 'C66', 'C67', 'C68', 'C69', 'C15', 'C25', 'C45', 'C55', 'C35', 'C75', 'C85', 'C95', 'C44', 'C46', 'C54', 'C56'],
                'C66':['C62', 'C63', 'C64', 'C65', 'C61', 'C67', 'C68', 'C69', 'C16', 'C26', 'C46', 'C56', 'C36', 'C76', 'C86', 'C96', 'C44', 'C45', 'C54', 'C55'],
                'C67':['C62', 'C63', 'C64', 'C65', 'C66', 'C61', 'C68', 'C69', 'C17', 'C27', 'C47', 'C57', 'C37', 'C77', 'C87', 'C97', 'C48', 'C49', 'C58', 'C59'],
                'C68':['C62', 'C63', 'C64', 'C65', 'C66', 'C67', 'C61', 'C69', 'C18', 'C28', 'C48', 'C58', 'C38', 'C78', 'C88', 'C98', 'C47', 'C49', 'C57', 'C59'],
                'C69':['C62', 'C63', 'C64', 'C65', 'C66', 'C67', 'C68', 'C61', 'C19', 'C29', 'C49', 'C59', 'C39', 'C79', 'C89', 'C99', 'C47', 'C48', 'C57', 'C58'],
                'C71':['C72', 'C73', 'C74', 'C75', 'C76', 'C77', 'C78', 'C79', 'C11', 'C21', 'C31', 'C51', 'C61', 'C41', 'C81', 'C91', 'C82', 'C83', 'C92', 'C93'],
                'C72':['C71', 'C73', 'C74', 'C75', 'C76', 'C77', 'C78', 'C79', 'C12', 'C22', 'C32', 'C52', 'C62', 'C42', 'C82', 'C92', 'C81', 'C83', 'C91', 'C93'],
                'C73':['C72', 'C71', 'C74', 'C75', 'C76', 'C77', 'C78', 'C79', 'C13', 'C23', 'C33', 'C53', 'C63', 'C43', 'C83', 'C93', 'C81', 'C82', 'C91', 'C92'],
                'C74':['C72', 'C73', 'C71', 'C75', 'C76', 'C77', 'C78', 'C79', 'C14', 'C24', 'C34', 'C54', 'C64', 'C44', 'C84', 'C94', 'C85', 'C86', 'C95', 'C96'],
                'C75':['C72', 'C73', 'C74', 'C71', 'C76', 'C77', 'C78', 'C79', 'C15', 'C25', 'C35', 'C55', 'C65', 'C45', 'C85', 'C95', 'C84', 'C86', 'C94', 'C96'],
                'C76':['C72', 'C73', 'C74', 'C75', 'C71', 'C77', 'C78', 'C79', 'C16', 'C26', 'C36', 'C56', 'C66', 'C46', 'C86', 'C96', 'C84', 'C85', 'C94', 'C95'],
                'C77':['C72', 'C73', 'C74', 'C75', 'C76', 'C71', 'C78', 'C79', 'C17', 'C27', 'C37', 'C57', 'C67', 'C47', 'C87', 'C97', 'C88', 'C89', 'C98', 'C99'],
                'C78':['C72', 'C73', 'C74', 'C75', 'C76', 'C77', 'C71', 'C79', 'C18', 'C28', 'C38', 'C58', 'C68', 'C48', 'C88', 'C98', 'C87', 'C89', 'C97', 'C99'],
                'C79':['C72', 'C73', 'C74', 'C75', 'C76', 'C77', 'C78', 'C71', 'C19', 'C29', 'C39', 'C59', 'C69', 'C49', 'C89', 'C99', 'C87', 'C88', 'C97', 'C98'],
                'C81':['C82', 'C83', 'C84', 'C85', 'C86', 'C87', 'C88', 'C89', 'C11', 'C31', 'C41', 'C21', 'C61', 'C71', 'C51', 'C91', 'C72', 'C73', 'C92', 'C93'],
                'C82':['C81', 'C83', 'C84', 'C85', 'C86', 'C87', 'C88', 'C89', 'C12', 'C32', 'C42', 'C22', 'C62', 'C72', 'C52', 'C92', 'C71', 'C73', 'C91', 'C93'],
                'C83':['C82', 'C81', 'C84', 'C85', 'C86', 'C87', 'C88', 'C89', 'C13', 'C33', 'C43', 'C23', 'C63', 'C73', 'C53', 'C93', 'C71', 'C72', 'C91', 'C92'],
                'C84':['C82', 'C83', 'C81', 'C85', 'C86', 'C87', 'C88', 'C89', 'C14', 'C34', 'C44', 'C24', 'C64', 'C74', 'C54', 'C94', 'C75', 'C76', 'C95', 'C96'],
                'C85':['C82', 'C83', 'C84', 'C81', 'C86', 'C87', 'C88', 'C89', 'C15', 'C35', 'C45', 'C25', 'C65', 'C75', 'C55', 'C95', 'C74', 'C76', 'C94', 'C96'],
                'C86':['C82', 'C83', 'C84', 'C85', 'C81', 'C87', 'C88', 'C89', 'C16', 'C36', 'C46', 'C26', 'C66', 'C76', 'C56', 'C96', 'C74', 'C75', 'C94', 'C95'],
                'C87':['C82', 'C83', 'C84', 'C85', 'C86', 'C81', 'C88', 'C89', 'C17', 'C37', 'C47', 'C27', 'C67', 'C77', 'C57', 'C97', 'C78', 'C79', 'C98', 'C99'],
                'C88':['C82', 'C83', 'C84', 'C85', 'C86', 'C87', 'C81', 'C89', 'C18', 'C38', 'C48', 'C28', 'C68', 'C78', 'C58', 'C98', 'C77', 'C79', 'C97', 'C99'],
                'C89':['C82', 'C83', 'C84', 'C85', 'C86', 'C87', 'C88', 'C81', 'C19', 'C39', 'C49', 'C29', 'C69', 'C79', 'C59', 'C99', 'C77', 'C78', 'C97', 'C98'],                
                'C91':['C92', 'C93', 'C94', 'C95', 'C96', 'C97', 'C98', 'C99', 'C11', 'C21', 'C41', 'C51', 'C31', 'C71', 'C81', 'C61', 'C72', 'C73', 'C82', 'C83'],
                'C92':['C91', 'C93', 'C94', 'C95', 'C96', 'C97', 'C98', 'C99', 'C12', 'C22', 'C42', 'C52', 'C32', 'C72', 'C82', 'C62', 'C71', 'C73', 'C81', 'C83'],
                'C93':['C92', 'C91', 'C94', 'C95', 'C96', 'C97', 'C98', 'C99', 'C13', 'C23', 'C43', 'C53', 'C33', 'C73', 'C83', 'C63', 'C71', 'C72', 'C81', 'C82'],
                'C94':['C92', 'C93', 'C91', 'C95', 'C96', 'C97', 'C98', 'C99', 'C14', 'C24', 'C44', 'C54', 'C34', 'C74', 'C84', 'C64', 'C75', 'C76', 'C85', 'C86'],
                'C95':['C92', 'C93', 'C94', 'C91', 'C96', 'C97', 'C98', 'C99', 'C15', 'C25', 'C45', 'C55', 'C35', 'C75', 'C85', 'C65', 'C74', 'C76', 'C84', 'C86'],
                'C96':['C92', 'C93', 'C94', 'C95', 'C91', 'C97', 'C98', 'C99', 'C16', 'C26', 'C46', 'C56', 'C36', 'C76', 'C86', 'C66', 'C74', 'C75', 'C84', 'C85'],
                'C97':['C92', 'C93', 'C94', 'C95', 'C96', 'C91', 'C98', 'C99', 'C17', 'C27', 'C47', 'C57', 'C37', 'C77', 'C87', 'C67', 'C78', 'C79', 'C88', 'C89'],
                'C98':['C92', 'C93', 'C94', 'C95', 'C96', 'C97', 'C91', 'C99', 'C18', 'C28', 'C48', 'C58', 'C38', 'C78', 'C88', 'C68', 'C77', 'C79', 'C87', 'C89'],
                'C99':['C92', 'C93', 'C94', 'C95', 'C96', 'C97', 'C98', 'C91', 'C19', 'C29', 'C49', 'C59', 'C39', 'C79', 'C89', 'C69', 'C77', 'C78', 'C87', 'C88']}

print("_______________________________________________________________")
#print(AC_3(csp))
#print(neighborsCSP)
#print(type(neighborsCSP))

#switch neighborsCSP9x9 to neighborsCSP4x4 with relevant data to use 4x4
csp = CSP(variablesCSP, domainsCSP, neighborsCSP9x9, constraintsCSP)
#print(revise(csp, 'C13', 'C21'))
#print(AC_3(csp))
#print(minimum_remaining_values(csp, {'C12', 'C34'}))
"""
run = True
while run == True:
    printme = backtracking_search(csp)
    if ((printme) == None):
        csp = CSP(variablesCSP, domainsCSP, neighborsCSP, constraintsCSP)
        continue
    else:
        run = False
"""
answer = backtracking_search(csp)
print("Answer to Sudoku puzzle:")
print(((answer)))
#works, but sometimes cells get switched so its weird to read;
#{'C11': 7, 'C14': 4, 'C15': 1, 'C12': 9, 'C13': 3 ... vs {'C11': 7, 'C12': 9, 'C13': 3, 'C14': 4, 'C15': 1 ...

f.close()

app = Flask(__name__)
@app.route('/')

@app.route("/")
def hello_world():
    returnme = []
    for a in answer:
        returnme.append(str(answer.get(a)))
    return returnme
